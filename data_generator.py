import tensorflow as tf
from tensorflow_addons.utils import types
from typeguard import typechecked
import numpy as np
import argparse
from scipy.stats import truncnorm
import ast
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable cuda sepeed up
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # disable CPU wornings


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", dest="output", default="naly_syn_simple_arith")
parser.add_argument("-d", "--dist", dest="dist", default="normal", help="Prob.Dist")
parser.add_argument(
    "-p", "--params", dest="params", default="(-3,3)", type=ast.literal_eval
)
parser.add_argument("-e", "--ext", dest="ext", default="(10,15)", type=ast.literal_eval)

parser.add_argument("-n", "--nalu", dest="nalu", default="nalui1")
parser.add_argument("-se", "--seed", dest="seed", default=42, type=int)
parser.add_argument("-op", "--operation", dest="op", default="MUL")


args = parser.parse_args("")


def sample(dist, params, numDim=3, numDP=64000):
    data = np.zeros(shape=(numDP, numDim))
    if dist == "normal":
        intmean = (params[0] + params[1]) / 2
        intstd = (params[1] - params[0]) / 6
        print(
            "Generating Data: \nInt: \tdist \t %s\n\t\tdata >=\t %s\n\t\tmean(s)\t %s\n\t\tdata <\t %s\n\t\tstd \t %s"
            % (dist, params[0], intmean, params[1], intstd)
        )
        mi, ma = (params[0] - intmean) / intstd, (params[1] - intmean) / intstd
        data = np.reshape(
            truncnorm.rvs(mi, ma, intmean, intstd, size=numDim * numDP), data.shape
        )

    elif dist == "uniform":
        print(
            "Generating Data: \nInt: \tdist \t %s\n\t\tdata >=\t %s\n\t\tdata <\t %s\n\t\t"
            % (dist, params[0], params[1])
        )
        data = np.reshape(
            np.random.uniform(params[0], params[1], size=numDim * numDP), data.shape
        )
    elif dist == "exponential":
        data = np.random.exponential(params, size=(numDP, numDim))
    else:
        raise Exception("Unknown distribution")
    data = np.reshape(data, [-1])  # reshape to mix both distributions per instance!
    np.random.shuffle(data)
    data = np.reshape(data, (numDP, numDim))
    return data


def operation(op, a, b):
    if op.lower() == "mul":
        return a * b
    if op.lower() == "add":
        return a + b
    if op.lower() == "sub":
        return a - b
    if op.lower() == "div":
        return a / b


def data_comb(data):
    return (data[:, 0] - data[:, 1]) * (data[:, 2] - data[:, 3]) + (
        data[:, 4] * data[:, 5]
    )


def get_data(input_dim, batch_size=None, ext=False):
    data = sample(args.dist, args.ext if ext else args.params, input_dim)
    lbls = data_comb(data)
    lbls = np.reshape(lbls, newshape=(-1, 1))
    data_dp = tf.data.Dataset.from_tensor_slices((data, lbls)).prefetch(
        tf.data.AUTOTUNE
    )
    if batch_size:
        data_dp = data_dp.batch(batch_size)
    return data_dp
