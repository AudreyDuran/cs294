"""
Algorithme DAager: entrainer policy a partir de verites terrains puis accumuler de nouvelles data viennent du run
Example usage:
    python3 DAgger.py expert_data/Hopper-v2_20rollouts.pkl Hopper-v2 --render --num_rollout 20
"""

import os
import pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tf_util
import gym
import load_policy
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_data_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--num_rollouts', type=int, default=20)
    parser.add_argument('--render', action='store_true')

    args = parser.parse_args()
