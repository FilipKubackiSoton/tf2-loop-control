import numpy as np
import tensorflow as tf
from iNALU import NALU
from data_generator import get_data
from typing import List
from loop_control import LoopControlerCallback


BATCH_SIZE = 64
INPUT_DIM = 7
data_dp = get_data(INPUT_DIM, BATCH_SIZE)


class LoopControlTest(tf.test.TestCase):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)

    def setUp(self):
        super().setUp()
        self.seed = 100
        # define model
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(INPUT_DIM),
                NALU(3, clipping=20),
                NALU(2, clipping=20),
                NALU(1, clipping=20),
            ]
        )

    def testLossFunctions(self):
        with self.session():
            tf.keras.backend.clear_session()
            tf.random.set_seed(self.seed)

            # compile model
            self.model.compile(
                optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01),
                loss=tf.keras.metrics.mean_absolute_error,  # "mse",
                metrics=["mae"],
            )

            def cond(self) -> bool:
                return self.epochs % 2 == 0

            config = {
                "gate": {
                    "cond": cond,
                    True: {"loss": tf.keras.metrics.mean_squared_error},
                    False: {},
                },
            }

            callback = LoopControlerCallback(config, {}, verbose=0)
            _ = self.model.fit(data_dp, epochs=4, verbose=0, callbacks=[callback])

            self.assertEqual(
                self.model.gate_on.get_concrete_function(
                    next(iter(data_dp))
                ).graph._names_in_use["squareddifference"],
                1
            )
            self.assertEqual(
                self.model.gate_on.get_concrete_function(
                    next(iter(data_dp))
                ).graph._names_in_use["mean"],
                2
            )

            self.assertIn(
                'mean',
                self.model.gate_off.get_concrete_function(
                    next(iter(data_dp))
                ).graph._names_in_use,
            )

            self.assertIn(
                'sub',
                self.model.gate_off.get_concrete_function(
                    next(iter(data_dp))
                ).graph._names_in_use,
            )

            self.assertIn(
                'abs',
                self.model.gate_off.get_concrete_function(
                    next(iter(data_dp))
                ).graph._names_in_use,
            )

    def testIncludVariables(self):
        with self.session():
            tf.keras.backend.clear_session()
            tf.random.set_seed(self.seed)

            # compile model
            self.model.compile(
                optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01),
                loss="mse",
                metrics=["mae"],
            )

            def cond(self) -> bool:
                return True

            def get_gates_variables(self) -> List[tf.Variable]:
                return [layer.g for layer in self.layers if isinstance(layer, NALU)]

            config = {
                "gate": {
                    "cond": cond,
                    True: {
                        "variables": get_gates_variables,
                    },
                },
            }

            callback = LoopControlerCallback(config, {}, verbose=0)
            _ = self.model.fit(data_dp, epochs=2, verbose=0, callbacks=[callback])
            priorAllVariables = {x.name: x.numpy() for x in callback._initAllVars}
            postAllVariables = {
                x.name: x.numpy() for x in self.model.trainable_variables
            }
            priorIncludVariables = callback._initIncludVars
            postIncludVariables = {
                n: postAllVariables[n] for n, _ in priorIncludVariables.items()
            }

            for var_name, var_value in priorIncludVariables.items():
                self.assertNotAllClose(var_value, postIncludVariables[var_name])

            for var_name, var_value in priorAllVariables.items():
                if var_name not in priorIncludVariables:
                    self.assertAllClose(var_value, postAllVariables[var_name])

    def testExcludVariables(self):
        with self.session():
            tf.keras.backend.clear_session()
            tf.random.set_seed(self.seed)

            # compile model
            self.model.compile(
                optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01),
                loss="mse",
                metrics=["mae"],
            )

            def cond(self) -> bool:
                return True

            def get_gates_variables(self) -> List[tf.Variable]:
                return [layer.g for layer in self.layers if isinstance(layer, NALU)]

            config = {
                "gate": {
                    "cond": cond,
                    True: {
                        "excluded_variables": get_gates_variables,
                    },
                },
            }

            callback = LoopControlerCallback(config, {}, verbose=0)
            _ = self.model.fit(data_dp, epochs=2, verbose=0, callbacks=[callback])
            priorAllVariables = {x.name: x.numpy() for x in callback._initAllVars}
            postAllVariables = {
                x.name: x.numpy() for x in self.model.trainable_variables
            }
            priorExcludVariables = callback._initExcludVars
            postExcludVariables = {
                n: postAllVariables[n] for n, _ in priorExcludVariables.items()
            }

            for var_name, var_value in priorExcludVariables.items():
                self.assertNotAllClose(var_value, postExcludVariables[var_name])

            for var_name, var_value in priorAllVariables.items():
                if var_name not in priorExcludVariables:
                    self.assertAllClose(var_value, postAllVariables[var_name])

    def testControlVariables(self):
        with self.session():
            tf.keras.backend.clear_session()
            tf.random.set_seed(self.seed)

            # compile model
            self.model.compile(
                optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01),
                loss="mse",
                metrics=["mae"],
            )

            def cond(self) -> bool:
                return self.epochs % 2 == 0

            config = {
                "gate": {
                    "cond": cond,
                    True: {},
                    False: {}
                },
            }

            callback = LoopControlerCallback(config, {}, verbose=0)
            history = self.model.fit(data_dp, epochs=2, verbose=0, callbacks=[callback])
            self.assertTrue(history.history["gate"][0])
            self.assertFalse(history.history["gate"][1])
            

    # def testClippingFunctions(self):
    #     with self.session():
    #         tf.keras.backend.clear_session()
    #         tf.random.set_seed(self.seed)

    #         # compile model
    #         self.model.compile(
    #             optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01),
    #             loss=tf.keras.metrics.mean_squared_error,  # "mse",
    #             metrics=["mae"],
    #         )

    #         def cond(self) -> bool:
    #             return self.epochs % 8 == 0

    #         config = {
    #             "gate": {
    #                 "cond": cond,
    #                 True: {
    #                     "clipping": (-0.1, 0.1),
    #                 },
    #                 False: {},
    #             },
    #         }

    #         callback = LoopControlerCallback(config, {}, verbose=0)
    #         _ = self.model.fit(data_dp, epochs=2, verbose=0, callbacks=[callback])
    #         self.assertIn(
    #             "clip_by_value",
    #             self.model.gate_on.get_concrete_function(
    #                 next(iter(data_dp))
    #             ).graph._names_in_use,
    #         )
    #         self.assertNotIn(
    #             "clip_by_value",
    #             self.model.gate_off.get_concrete_function(
    #                 next(iter(data_dp))
    #             ).graph._names_in_use,
    #         )


tf.test.main()
