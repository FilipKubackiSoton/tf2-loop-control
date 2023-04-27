import tensorflow as tf
from typing import List
import json
import os
from data_generator import get_data, get_test_data
from iNALU import NALU
from loop_control import LoopControlerCallback

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable cuda sepeed up
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # disable CPU wornings


BATCH_SIZE = 64
INPUT_DIM = 7
data_dp = get_data(INPUT_DIM, BATCH_SIZE)
ext_dp = get_test_data(INPUT_DIM, BATCH_SIZE, True)
int_dp = get_test_data(INPUT_DIM, BATCH_SIZE)

seeds = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
res = {}
res_h = {}
res_t = {}
for seed in seeds:
    print(f"-------------Seed: {seed}-------------")
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)

    # define model
    model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(INPUT_DIM),
            NALU(3, clipping=20),
            NALU(2, clipping=20),
            NALU(1, clipping=20)]
    )

    # compile model
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01),
        loss="mse",
        metrics=["mae"],
    )

    def gate_open(self) -> bool:
        return self.batches % 10 < 8

    def delay_reg(self) -> bool:
        return self.epochs > 10 and self.history and self.history[-1]["loss"] < 1.0

    def get_gates_variables(self) -> List[tf.Variable]:
        return [layer.g for layer in self.layers if isinstance(layer, NALU)]

    config = {
        "gate": {
            "cond": gate_open,
            True: {"clipping": (-0.1, 0.1), "excluded_variables": get_gates_variables},
            False: {
                "clipping": (-0.1, 0.1),
                "variables": get_gates_variables,
            },
        },
        "delay": {"cond": delay_reg, True: {"loss": False, "regularize": True}},
    }

    def reinit_cond(self):
        reinit_loss = [x["loss"] for x in self.history]
        return (len(reinit_loss) > 10000) and (
            tf.math.reduce_mean(reinit_loss[-3000:]) > 0.2
        )

    def reinitialise_fn(self):
        for layer in self.layers:
            if isinstance(layer, NALU):
                layer.reinitialize()

    def reinitialise_metrics(self):
        self.history = []
        self.epochs = 0
        self.batches = 0

    def early_stopping(self):
        return tf.experimental.numpy.log10(
            tf.math.reduce_mean([x["loss"] for x in self.history][-3000:])
            ) < -10
                
    def stop_training(self):
        self.model.stop_training = True

    async_config = {
        "reinitialize": {
            "cond": reinit_cond,
            "model": reinitialise_fn,
            "callback": reinitialise_metrics,
        },
        "eraly_stopping": {
            "cond": early_stopping,
            "callback": stop_training,
        }
    }

    class InExtrapolationEvaluationCallback(tf.keras.callbacks.Callback):
        def __init__(self, ext_data, int_data, *args, **kwargs):
            super(InExtrapolationEvaluationCallback, self).__init__(*args, **kwargs)
            self.ext_data, self.int_data = ext_data, int_data
            self.ext_results, self.int_results = {}, {}
            self.test_per_epochs = 1
            self.epochs = 0
                    
        def on_epoch_end(self, epochs, log=None):
            if self.epochs % self.test_per_epochs == 0:
                self.int_results[self.epochs] = tf.reduce_mean(tf.keras.losses.get("mse")(self.model.predict(self.int_data[0], verbose=0), self.int_data[1])).numpy()
                self.ext_results[self.epochs] = tf.reduce_mean(tf.keras.losses.get("mse")(self.model.predict(self.ext_data[0], verbose=0), self.ext_data[1])).numpy()
            self.epochs += 1

    loopControl = LoopControlerCallback(config, async_config, verbose=1)
    inExEvaluation = InExtrapolationEvaluationCallback(ext_dp, int_dp)
    history = model.fit(data_dp, epochs=120, verbose=1, callbacks=[
        loopControl,
        inExEvaluation
        ])

    res[seed] = history.history
    res_h[seed] = loopControl.history
    res_t[seed] = {
        "ext": inExEvaluation.exp_results,
        "int": inExEvaluation.int_results, 
    }

with open("results/result5.json", "w") as f:
    json.dump(res, f, indent=4)

with open("results/result5_h.json", "w") as f:
    json.dump(res_h, f, indent=4)

with open("results/result5_t.json", "w") as f:
    json.dump(res_t, f, indent=4)
