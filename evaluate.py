# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 00:04:46 2024

@author: 91216
"""

import os
gpu_num = "" # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sionna

from sionna.utils import BinarySource, ebnodb2no, log10, expand_to_rank, insert_dims

from sionna.utils import sim_ber

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
import pickle
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
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense
###############################################
# SNR range for evaluation and training [dB]
###############################################
ebno_db_min = 5.0
ebno_db_max = 6.0

###############################################
# Training configuration
###############################################
num_training_iterations_conventional = 1000 # Number of training iterations for conventional training
# Number of training iterations with RL-based training for the alternating training phase and fine-tuning of the receiver phase
num_training_iterations_rl_alt = 7000
num_training_iterations_rl_finetuning = 3000
training_batch_size = tf.constant(32, tf.int32) # Training batch size
rl_perturbation_var = 0.01 # Variance of the perturbation used for RL-based training of the transmitter
model_weights_path_conventional_training = "3gpp_autoencoder_weights_conventional_training" # Filename to save the autoencoder weights once conventional training is done
results_filename = "3gpp_autoencoder_results"
class Model(tf.keras.Model):
    def __init__(self,
                 scenario,    # "umi", "uma", "rma"
                 perfect_csi, # bool
                 domain,      # "freq", "time"
                 detector,    # "lmmse", "kbest"
                 speed,       # float
                 training
                ):
        super().__init__()
        self._scenario = scenario
        self._perfect_csi = perfect_csi
        self._domain = domain
        self._speed = speed
        self._training = training

        self._carrier_frequency = 3.5e9
        self._subcarrier_spacing = 30e3
        self._num_tx = 1
        self._num_tx_ant = 2
        self._num_layers = 1
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
        constellation = self._pusch_transmitter.constellation
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
                                      constellation=constellation,
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
        
        #################
        # Loss function
        #################
        if self._training:
            self._bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function(jit_compile=True)
    def call(self, batch_size, ebno_db):
        self.new_topology(batch_size)
        x, b, c = self._pusch_transmitter(batch_size)
        no = ebnodb2no(ebno_db,
                       self._pusch_transmitter._num_bits_per_symbol,
                       self._pusch_transmitter._target_coderate,
                       self._pusch_transmitter.resource_grid)
        y, h = self._channel([x, no])
        if self._perfect_csi:
                b_hat, llr = self._pusch_receiver([y, h, no])
        else:
                b_hat, llr = self._pusch_receiver([y, h, no])
        if self._training:
               # print(f"c shape: {c.shape}, c type: {type(c)}")
                #print(f"llr shape: {llr.shape}, llr type: {type(llr)}")
                # 假设需要将它们都转换为 2D Tensor，具体的形状取决于你的任务
                c = tf.reshape(c, [batch_size, -1])
                llr = tf.reshape(llr, [batch_size, -1])

                loss = self._bce(c, llr)
                #print('loss1', loss)  # 检查返回值是如何构成的
                # 将多个损失值取平均，得到一个标量 Tensor
                loss = tf.reduce_mean(loss)
                #print('loss2', loss)  # 检查返回值是如何构成的

                #print(type(loss))
                return loss, b, b_hat
        else:
                return b, b_hat

def load_weights(model, model_weights_path):
    #model(1, tf.constant(10.0, tf.float32))
    #with open(model_weights_path, 'rb') as f:
        #weights = pickle.load(f)
    #model.set_weights(weights)
# Dictionary storing the results
# 根据模型的输入调整 dummy_input
    dummy_input = tf.constant(32, dtype=tf.int32) # 1 为 batch_size，32 为假设的输入维度
    dummy_ebno_db = tf.constant([10.0], dtype=tf.float32)
    model(dummy_input, dummy_ebno_db)  # 预热模型
    with open(model_weights_path, 'rb') as f:
        weights = pickle.load(f)
    model.set_weights(weights)
BLER = {}
ebno_dbs = np.arange(ebno_db_min, # Min SNR for evaluation
                     ebno_db_max, # Max SNR for evaluation
                     1) # Step


model_conventional = Model(scenario="umi", domain= "freq", perfect_csi= True, detector= "lmmse", speed=3.0, training=False)
load_weights(model_conventional, model_weights_path_conventional_training)
fig = model_conventional._pusch_transmitter.constellation.show()
fig.suptitle('Conventional training');
#fig = model_conventional._pusch_transmitter.constellation.show()
_,bler = sim_ber(model_conventional, ebno_dbs, batch_size=32, num_target_block_errors=200, max_mc_iter=1000)
#_,bler = sim_ber(model_conventional, ebno_dbs, batch_size=128, num_target_block_errors=1000, max_mc_iter=500)
BLER['autoencoder-conv'] = bler.numpy()


with open(results_filename, 'wb') as f:
    pickle.dump((ebno_dbs, BLER), f)
plt.figure(figsize=(10,8))
# Baseline - Perfect CSI
#plt.semilogy(ebno_dbs, BLER['baseline'], 'o-', c=f'C0', label=f'Baseline')
# Autoencoder - conventional training
plt.semilogy(ebno_dbs, BLER['autoencoder-conv'], 'x-.', c=f'C1', label=f'conventional training')

plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.grid(which="both")
plt.ylim((1e-4, 1.0))
plt.legend()
plt.tight_layout()

plt.show()

model_conventional = Model(scenario="umi", domain= "freq", perfect_csi= True, detector= "lmmse", speed=3.0, training=True)
print("Before loading weights:", model_conventional._pusch_transmitter.constellation.points.numpy())
fig = model_conventional._pusch_transmitter.constellation.show()
fig.suptitle('constellation before traning');
load_weights(model_conventional, model_weights_path_conventional_training)
print("After loading weights:", model_conventional._pusch_transmitter.constellation.points.numpy())
fig = model_conventional._pusch_transmitter.constellation.show()
fig.suptitle('Conventional training');