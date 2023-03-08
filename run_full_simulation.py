import tensorflow as tf
from typing import Dict, Any, List, Tuple
import functools
from tensorflow_addons.utils import types
from typeguard import typechecked
import numpy as np
import argparse
from scipy.stats import truncnorm
import ast
import json
import os
from data_generator import get_data
from iNALU import NALU
from loop_control import LoopControlerCallback
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable cuda sepeed up
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # disable CPU wornings


BATCH_SIZE = 64
INPUT_DIM = 7
data_dp = get_data(INPUT_DIM, BATCH_SIZE)

seeds = [100, 101 ] #, 102, 103, 104, 105, 106, 107, 108, 109]
res = {}
res_h = {}
for seed in seeds:
    print(f"-------------Seed: {seed}-------------")
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(INPUT_DIM),
            NALU(3, clipping=20),
            NALU(2, clipping=20),
            NALU(1, clipping=20),
        ]
    )

    # compile model
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01),
        loss="mse",
        metrics=["mae"],
    )

    def gate_open(self):
        return self.batches % 10 < 8

    def delay_reg(self):
        return (
            self.reinit_epochs > 10
            and self.reinit_history
            and self.reinit_history[-1]["loss"] < 1.0
        )

    def get_gates_variables(self) -> List[tf.Variable]:
        return [l.g for l in self.layers if isinstance(l, NALU)]

    config = {
        "gate": {
            "cond": gate_open,
            True: {"clipping": (-0.1, 0.1),
                   "excluded_variables": get_gates_variables
                   },
            False: {
                "clipping": (-0.1, 0.1),
                "variables": get_gates_variables,
            },
        },
        "delay": {"cond": delay_reg, True: {"loss": False, "regularize": True}},
    }

    def reinit_cond(self):
        # reinit_loss = [x["loss"] for x in self.reinit_history]
        # # subset_index = len(self.reinit_history)//2
        # split_index = len(self.reinit_history)//2
        # return (len(reinit_loss) > 10000) and np.mean(reinit_loss[split_index:]) > 1 and np.mean(reinit_loss[:split_index]) + np.std(reinit_loss[:split_index]) <= np.mean(reinit_loss[split_index:])
        split_index = len(self.reinit_history) // 2
        reinit_loss = [x["loss"] for x in self.reinit_history]
        return (len(reinit_loss) > 10000) and (
            tf.math.reduce_mean(reinit_loss[-3000:]) > 0.2
        )

    def reinitialize_fn(self):
        for l in self.layers:
            if isinstance(l, NALU):
                l.reinitialize()

    reinit_config = {
        "cond": reinit_cond,
        "reinit_fn": reinitialize_fn,
    }

    lcc = LoopControlerCallback(config, reinit_config, verbose=0)
    history = model.fit(data_dp, epochs=120, verbose=1, callbacks=[lcc])
    res[seed] = history.history
    res_h[seed] = lcc.history

with open("results/result2.json", "w") as f:
    json.dump(res, f, indent=4)

with open("results/result2_h.json", "w") as f:
    json.dump(res_h, f, indent=4)
