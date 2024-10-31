# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:37:28 2024

@author: 91216
"""
import os
gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import sionna
except ImportError as e:
    # Install Sionna if package is not already installed
    import os
    os.system("pip install sionna")
    import sionna

# Load the required Sionna components
from sionna.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.channel import AWGN, RayleighBlockFading, OFDMChannel, TimeChannel, time_lag_discrete_time_channel
from sionna.channel.tr38901 import AntennaArray, UMi, UMa, RMa
from sionna.channel import gen_single_sector_topology as gen_topology
from sionna.utils import compute_ber, ebnodb2no, sim_ber
from sionna.ofdm import KBestDetector, LinearDetector
from sionna.mimo import StreamManagement 
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf
# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

# We need to enable sionna.config.xla_compat before we can use
# tf.function with jit_compile=True.
# See https://nvlabs.github.io/sionna/api/config.html#sionna.Config.xla_compat
sionna.config.xla_compat=True

class Model(tf.keras.Model):
    """Simulate PUSCH transmissions over a 3GPP 38.901 model.

    This model runs BER simulations for a multiuser MIMO uplink channel
    compliant with the 5G NR PUSCH specifications.
    You can pick different scenarios, i.e., channel models, perfect or
    estimated CSI, as well as different MIMO detectors (LMMSE or KBest).
    You can chosse to run simulations in either time ("time") or frequency ("freq")
    domains and configure different user speeds.

    Parameters
    ----------
    scenario : str, one of ["umi", "uma", "rma"]
        3GPP 38.901 channel model to be used

    perfect_csi : bool
        Determines if perfect CSI is assumed or if the CSI is estimated

    domain :  str, one of ["freq", "time"]
        Domain in which the simulations are carried out.
        Time domain modelling is typically more complex but allows modelling
        of realistic effects such as inter-symbol interference of subcarrier
        interference due to very high speeds.

    detector : str, one of ["lmmse", "kbest"]
        MIMO detector to be used. Note that each detector has additional
        parameters that can be configured in the source code of the _init_ call.

    speed: float
        User speed (m/s)

    Input
    -----
    batch_size : int
        Number of simultaneously simulated slots

    ebno_db : float
        Signal-to-noise-ratio

    Output
    ------
    b : [batch_size, num_tx, tb_size], tf.float
        Transmitted information bits

    b_hat : [batch_size, num_tx, tb_size], tf.float
        Decoded information bits
    """
    def __init__(self,
                 scenario,    # "umi", "uma", "rma"
                 perfect_csi, # bool
                 domain,      # "freq", "time"
                 detector,    # "lmmse", "kbest"
                 speed        # float
                ):
        super().__init__()
        self._scenario = scenario
        self._perfect_csi = perfect_csi
        self._domain = domain
        self._speed = speed

        self._carrier_frequency = 3.5e9
        self._subcarrier_spacing = 30e3
        self._num_tx = 1
        #self._num_tx = 1
        self._num_tx_ant = 2
        #self._num_tx_ant = 2
        self._num_layers = 1
        #self._num_layers = 1
        self._num_rx_ant = 16
        self._mcs_index = 17
        self._mcs_table = 1
        self._num_prb = 16

        # Create PUSCHConfigs

        # PUSCHConfig for the first transmitter
        pusch_config = PUSCHConfig()
        pusch_config.carrier.subcarrier_spacing = self._subcarrier_spacing/1000
        pusch_config.carrier.n_size_grid = self._num_prb
        pusch_config.num_antenna_ports = self._num_tx_ant
        pusch_config.num_layers = self._num_layers
        pusch_config.precoding = "codebook"
        pusch_config.tpmi = 1
        pusch_config.dmrs.dmrs_port_set = list(range(self._num_layers))
        pusch_config.dmrs.config_type = 2
        pusch_config.dmrs.length = 2
        pusch_config.dmrs.additional_position = 1
        pusch_config.dmrs.num_cdm_groups_without_data = 3
        pusch_config.tb.mcs_index = self._mcs_index
        pusch_config.tb.mcs_table = self._mcs_table

        # Create PUSCHConfigs for the other transmitters by cloning of the first PUSCHConfig
        # and modifying the used DMRS ports.
        pusch_configs = [pusch_config]
        for i in range(1, self._num_tx):
            pc = pusch_config.clone()
            pc.dmrs.dmrs_port_set = list(range(i*self._num_layers, (i+1)*self._num_layers))
            pusch_configs.append(pc)

        # Create PUSCHTransmitter
        self._pusch_transmitter = PUSCHTransmitter(pusch_configs, output_domain=self._domain)

        # Create PUSCHReceiver
        self._l_min, self._l_max = time_lag_discrete_time_channel(self._pusch_transmitter.resource_grid.bandwidth)


        rx_tx_association = np.ones([1, self._num_tx], bool)
        stream_management = StreamManagement(rx_tx_association,
                                             self._num_layers)

        assert detector in["lmmse", "kbest"], "Unsupported MIMO detector"
        if detector=="lmmse":
            detector = LinearDetector(equalizer="lmmse",
                                      output="bit",
                                      demapping_method="maxlog",
                                      resource_grid=self._pusch_transmitter.resource_grid,
                                      stream_management=stream_management,
                                      constellation_type="qam",
                                      num_bits_per_symbol=pusch_config.tb.num_bits_per_symbol)
        elif detector=="kbest":
            detector = KBestDetector(output="bit",
                                     num_streams=self._num_tx*self._num_layers,
                                     k=64,
                                     resource_grid=self._pusch_transmitter.resource_grid,
                                     stream_management=stream_management,
                                     constellation_type="qam",
                                     num_bits_per_symbol=pusch_config.tb.num_bits_per_symbol)

        if self._perfect_csi:
            self._pusch_receiver = PUSCHReceiver(self._pusch_transmitter,
                                                 mimo_detector=detector,
                                                 input_domain=self._domain,
                                                 channel_estimator="perfect",
                                                 l_min = self._l_min)
        else:
            self._pusch_receiver = PUSCHReceiver(self._pusch_transmitter,
                                                 mimo_detector=detector,
                                                 input_domain=self._domain,
                                                 l_min = self._l_min)

        # Configure antenna arrays
        self._ut_array = AntennaArray(
                                 num_rows=1,
                                 num_cols=int(self._num_tx_ant/2),
                                 polarization="dual",
                                 polarization_type="cross",
                                 antenna_pattern="omni",
                                 carrier_frequency=self._carrier_frequency)

        self._bs_array = AntennaArray(num_rows=1,
                                      num_cols=int(self._num_rx_ant/2),
                                      polarization="dual",
                                      polarization_type="cross",
                                      antenna_pattern="38.901",
                                      carrier_frequency=self._carrier_frequency)

        # Configure the channel model
        if self._scenario == "umi":
            self._channel_model = UMi(carrier_frequency=self._carrier_frequency,
                                      o2i_model="low",
                                      ut_array=self._ut_array,
                                      bs_array=self._bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
        elif self._scenario == "uma":
            self._channel_model = UMa(carrier_frequency=self._carrier_frequency,
                                      o2i_model="low",
                                      ut_array=self._ut_array,
                                      bs_array=self._bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
        elif self._scenario == "rma":
            self._channel_model = RMa(carrier_frequency=self._carrier_frequency,
                                      ut_array=self._ut_array,
                                      bs_array=self._bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)

        # Configure the actual channel
        if domain=="freq":
            self._channel = OFDMChannel(
                                self._channel_model,
                                self._pusch_transmitter.resource_grid,
                                normalize_channel=True,
                                return_channel=True)
        else:
            self._channel = TimeChannel(
                                self._channel_model,
                                self._pusch_transmitter.resource_grid.bandwidth,
                                self._pusch_transmitter.resource_grid.num_time_samples,
                                l_min=self._l_min,
                                l_max=self._l_max,
                                normalize_channel=True,
                                return_channel=True)

    def new_topology(self, batch_size):
        """Set new topology"""
        topology = gen_topology(batch_size,
                                self._num_tx,
                                self._scenario,
                                min_ut_velocity=self._speed,
                                max_ut_velocity=self._speed)

        self._channel_model.set_topology(*topology)

    #@tf.function(jit_compile=True)
    def call(self, batch_size, ebno_db):
        self.new_topology(batch_size)
        x, b, _ = self._pusch_transmitter(batch_size)
        no = ebnodb2no(ebno_db,
                       self._pusch_transmitter._num_bits_per_symbol,
                       self._pusch_transmitter._target_coderate,
                       self._pusch_transmitter.resource_grid)
        y, h = self._channel([x, no])
        #print(self._pusch_transmitter._num_bits_per_symbol)
        if self._perfect_csi:
            b_hat, _ = self._pusch_receiver([y, h, no])
        else:
            b_hat, _ = self._pusch_receiver([y, no])
        return b, b_hat
PUSCH_SIMS = {
    "scenario" : ["umi"],
    "domain" : ["freq"],
    #"perfect_csi" : [True, False],
    "perfect_csi" : [True],
    #"detector" : ["kbest", "lmmse"],
    "detector" : ["lmmse"],
    #"ebno_db" : list(range(0,8)),
    "ebno_db" : list(np.arange(5, 6, 1)),
    "speed" : 3.0,
    "batch_size_freq" : 32,
    "batch_size_time" : 28, # Reduced batch size from time-domain modeling
    "bler" : [],
    "ber" : []
    }

start = time.time()

for scenario in PUSCH_SIMS["scenario"]:
    for domain in PUSCH_SIMS["domain"]:
        for perfect_csi in PUSCH_SIMS["perfect_csi"]:
            batch_size = PUSCH_SIMS["batch_size_freq"] if domain=="freq" else PUSCH_SIMS["batch_size_time"]
            for detector in PUSCH_SIMS["detector"]:
                model = Model(scenario, perfect_csi, domain, detector, PUSCH_SIMS["speed"])
                ber, bler = sim_ber(model,
                            PUSCH_SIMS["ebno_db"],
                            batch_size=batch_size,
                            max_mc_iter=1500,
                            num_target_block_errors=50)
                PUSCH_SIMS["ber"].append(list(ber.numpy()))
                PUSCH_SIMS["bler"].append(list(bler.numpy()))

PUSCH_SIMS["duration"] = time.time() - start



'''# Load results (un-comment to show saved results from the cell above)
PUSCH_SIMS = eval("{'scenario': ['umi'], 'domain': ['freq'], 'perfect_csi': [True, False], 'detector': ['kbest', 'lmmse'], 'ebno_db': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'speed': 3.0, 'batch_size_freq': 128, 'batch_size_time': 28, 'bler': [[0.865234375, 0.525390625, 0.236328125, 0.0703125, 0.022894965277777776, 0.0081787109375, 0.0031916920731707315, 0.0011800130208333333, 0.0007274208566108007, 0.000298828125, 0.000189453125, 0.000107421875, 4.6875e-05], [0.501953125, 0.2412109375, 0.13411458333333334, 0.06556919642857142, 0.037109375, 0.021073190789473683, 0.012251420454545454, 0.007244318181818182, 0.0038869121287128713, 0.0027646346830985913, 0.0015751008064516128, 0.0009838684538653367, 0.0007239105504587156], [0.994140625, 0.97265625, 0.849609375, 0.513671875, 0.234375, 0.115234375, 0.05126953125, 0.028878348214285716, 0.023092830882352942, 0.018694196428571428, 0.013671875, 0.012451171875, 0.013739224137931034], [0.919921875, 0.724609375, 0.533203125, 0.2802734375, 0.16536458333333334, 0.08984375, 0.056919642857142856, 0.043619791666666664, 0.035006009615384616, 0.02055921052631579, 0.017578125, 0.018465909090909092, 0.01532451923076923]], 'ber': [[0.08414149284362793, 0.03808903694152832, 0.013622879981994629, 0.00516200065612793, 0.0018940899107191297, 0.0006881306568781534, 0.0002859627328267912, 0.00012890001138051352, 9.374645169220823e-05, 3.3643484115600584e-05, 2.6883602142333983e-05, 1.2900114059448242e-05, 6.676435470581055e-06], [0.032366275787353516, 0.015960693359375, 0.009874105453491211, 0.004354306629725865, 0.00270201943137429, 0.0015692459909539473, 0.0008932893926447088, 0.0005442922765558416, 0.0002903820264457476, 0.00021878598441540356, 0.00013986518306116904, 7.878217911185171e-05, 6.43913898992976e-05], [0.15517663955688477, 0.12702298164367676, 0.08907222747802734, 0.036322832107543945, 0.015564680099487305, 0.008847415447235107, 0.005330264568328857, 0.003527675356183733, 0.0029088469112620633, 0.0025938579014369418, 0.002038750155218716, 0.0017822608351707458, 0.001927071604235419], [0.10343790054321289, 0.06611466407775879, 0.043680429458618164, 0.0217667818069458, 0.013199090957641602, 0.007306861877441406, 0.005208117621285575, 0.004094309277004666, 0.003994941711425781, 0.002383282310084293, 0.0023060985233472743, 0.002356225794011896, 0.002158962763272799]], 'duration': 4399.180883407593}")
print("Simulation duration: {:1.2f} [h]".format(PUSCH_SIMS["duration"]/3600))

plt.figure()
plt.title("5G NR PUSCH over UMi Channel Model (8x16)")
plt.xlabel("SNR (dB)")
plt.ylabel("BLER")
plt.grid(which="both")
plt.xlim([PUSCH_SIMS["ebno_db"][0], PUSCH_SIMS["ebno_db"][-1]])
plt.ylim([1e-5, 1.0])

i = 0
legend = []
for scenario in PUSCH_SIMS["scenario"]:
    for domain in PUSCH_SIMS["domain"]:
        for perfect_csi in PUSCH_SIMS["perfect_csi"]:
            for detector in PUSCH_SIMS["detector"]:
                plt.semilogy(PUSCH_SIMS["ebno_db"], PUSCH_SIMS["bler"][i])
                i += 1
                csi = "Perf. CSI" if perfect_csi else "Imperf. CSI"
                det = "K-Best" if detector=="kbest" else "LMMSE"
                legend.append(det + " " + csi)
plt.legend(legend);'''