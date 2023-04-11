import tensorflow as tf
from typing import List
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

seeds = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
res = {}
res_h = {}
for seed in seeds:
    print(f"-------------Seed: {seed}-------------")
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)

    import tensorflow as tf
    from iNALU import NALU

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

    def reinitialize_fn(self):
        for layer in self.layers:
            if isinstance(layer, NALU):
                layer.reinitialize()

    def reinitialize_metrics(self):
        self.history = []
        self.epochs = 0
        self.batches = 0

    def early_stopping(self):
        return tf.experimental.numpy.log10(tf.math.reduce_mean([x["loss"] for x in self.history][-3000:])) < -10
                
    def stop_training(self):
        self.model.stop_training = True

    async_config = {
        "reinitialize": {
            "cond": reinit_cond,
            "model": reinitialize_fn,
            "callback": reinitialize_metrics,
        },
        "eraly_stopping": {
            "cond": early_stopping,
            "callback": stop_training,
        }
    }

    lcc = LoopControlerCallback(config, async_config, verbose=1)
    history = model.fit(data_dp, epochs=2, verbose=1, callbacks=[lcc])

    res[seed] = history.history
    res_h[seed] = lcc.history


# def async_foo_cond(self):
#     # implement your logic
#     return True

# def async_foo_fn_positive(self):
#     # bind to the model instance
#     return
    
# def async_foo_fn_negative(self):
#     # bind to the model instance
#     return

# async_config = {
#     "foo": {
#         "cond": async_foo_cond,
#         True : async_foo_fn_positive,
#         False: async_foo_fn_negative

#     }
# }

# # reinit_loss = [x["loss"] for x in self.reinit_history]
# # # subset_index = len(self.reinit_history)//2
# # split_index = len(self.reinit_history)//2
# # return (len(reinit_loss) > 10000) and np.mean(reinit_loss[split_index:]) > 1 and np.mean(reinit_loss[:split_index]) + np.std(reinit_loss[:split_index]) <= np.mean(reinit_loss[split_index:])
# split_index = len(self.reinit_history) // 2


# with open("results/result3.json", "w") as f:
#     json.dump(res, f, indent=4)

# with open("results/result3_h.json", "w") as f:
#     json.dump(res_h, f, indent=4)

# foo_cond = ""

# def get_gates_variables(self) -> List[tf.Variable]:
#     """Iterate through layers and extract gate
#     variables from each instance of NALU layer.

#     Returns:
#         List[tf.Variable]: array of gate variables
#     """
#     return [layer.g for layer in self.layers
#             if isinstance(layer, NALU)]

# config = {
#     "gate": {
#         "cond": gate_open,
#         True: {
#             "variables": get_gates_variables,
#             "excluded_variables": get_gates_variables


#             },
#     },
# }


# config = {
#     "foo": {
#         "cond": foo_cond,
#         True: {
#             "clipping": (-0.1, 0.1)

#         },
#     },
# }

# self = ""
# grads = ""
# tape = None

# self.optimizer.apply_gradients(zip(
#     [tf.clip_by_value(g, -0.1, 0.1) for g in grads],
#     tape.watched_variables()))

# action_1_config, action_2_config = "", ""

# config = {
#     # dictionary holding training configuration
#     # for steps taken in action_1
#     "action_1": action_1_config,
#     # dictionary holding training configuration
#     # for steps taken in action_2
#     "action_2": action_2_config,
#     # more actions
#     }

# action_1_config_true=""
# action_1_config_false=""


# def cond_function(self) -> bool:
#     # implement your custom logic
#     return True


# action_1_config = {
#     "cond": cond_function,
#     True: action_1_config_true,
#     False: action_1_config_false
# }


# get_variables = ""

# action_1_config_true = {
#     "loss": tf.keras.losses.Loss()/False/None,
#     "clipping": (-0.1, 0.1),
#     "variables": get_variables,
#     "excluded_variables": get_variables,
#     "regularize": True/False/None
# }


# from typing import Dict, Any

# def bind(a, b):
#     pass

# def _bind_slave_step(self,
#                      action_name: str,
#                      fn_config: Dict[str, Any],
#                      branch: bool) -> str:
#         """Bind train_substep method to the model instance.
#         It is model internal funciton that is called by
#         the train_step. This step implements the whole
#         logic related to the model's weight updates.

#         Examples:
#             ....

#         Args:
#             action_name (str): action name of a slave_step
#             fn_config (Dict[str, Any]): Configuration for train_substep
#             interpreted from default configuration file passed as the
#             argumnet of the constructor.
#             branch (bool): predicate if we generate train_substep for
#             True or False evaluation of the slave step.
#         """
#         lscope = {**locals(), **fn_config}
#         fn_name = self._get_actoin_step_name(action_name, branch)

#         function_body = f"""
# @tf.function
# def {fn_name}(self, data):
#     """
#         # graidnet tape configuration

#         function_body = f"""
#     logits = self(x, training=True)
#     loss_value = {'tf.constant(0, dtype=tf.float32) * '
#         if fn_config["loss"]==False else ''} loss(y, logits)
#     """

#         # remaining part of the function body

#         exec(function_body, {**globals(), **lscope}, lscope)
#         bind(self.model, lscope[fn_name])

#         return function_body


# _bind_slave_step(None, None, None, None)
