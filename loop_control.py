import tensorflow as tf
from typing import Dict, Any, List

def bind(instance, func, as_name=None):
    """
    Bind the function *func* to *instance*, with either provided name *as_name*
    or the existing name of *func*. The provided *func* should accept the
    instance as the first argument, i.e. "self".
    """
    if as_name is None:
        as_name = func.__name__
    bound_method = func.__get__(instance, instance.__class__)
    setattr(instance, as_name, bound_method)
    return bound_method


class LoopControlerCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        config: int,
        reinit_config: Dict[str, Any] = None,
        default_in_branch: Dict[str, Any] = {
            "loss": True,
            "regularize": None,
            "clipping": None,
            "variables": None,
            "excluded_variables": None,
        },
        verbose: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super(LoopControlerCallback, self).__init__(*args, **kwargs)
        self.default_in_branch: Dict[str, Any] = default_in_branch
        self.config: Dict[str, Any] = config
        self.reinit_config = reinit_config
        self.verbose: bool = verbose

    def on_train_begin(self, logs=None):
        """Function called directely before training. It is executed on each call of model.fit with this callback.
            Inside the scope of this funciton we can access model on which this callback works: self.model.

        Args:
            logs (_type_, optional): _description_. Defaults to None.
        """

        # meta attributes for building conditions
        self.epochs: int = 0
        self.batches: int = 0
        self.history: List[Any] = []

        # meta attributes for reinitialization handeling
        self.reinit_epochs: int = 0
        self.reinit_batches: int = 0
        self.reinit_history: List[Any] = []
        self.reinit_current_history: List[Any] = []

        # extend config with validation step
        self.config = self._extend_config(self.config)

        # assign active and exluded variables for slave steps
        self._extract_substeps_varialbe_arrays(self.config)

        # bind control variables and control conditions
        self._bind_controlers(self.config)

        # bind master train_step to model
        self._bind_master_step(self.config)

        # bind slave train_steps to model
        self._bind_slaves_steps(self.config)

        # bind reinitalization functionality to model
        # only if reinitalizatoin config is provided.
        # By default reinitialization condition is evaluated
        # at the end of each training epoch.
        if self.reinit_config:
            self._bind_reinitialization(self.reinit_config)

        # test
        self.model.optimizer.build(self.model.trainable_variables)

    def _bind_master_step(self, config: Dict[str, Any]) -> None:
        """Bind train_step method to the model instance.
        It is model internal funciton that is called for
        each train step inside tf.fit(...) scope.
        This steps constitutes compbination of smaller
        training substeps that we call slaves steps.

        Examples:
            config = {"gate": { "cond": gate_open,
                True: {"clipping": (-0.1, 0.1)}, False: {}},
            "delay": {"cond": delay_reg,
                True: {"loss": False, "regularize": True}}}
            _bind_master_step(config) -> bind train_step function below

            @tf.function
            def train_step(self, data):
                slave_loss = {'loss_gate' : tf.cond(self.control_variables['gate'],
                        lambda: self.gate_on(data), lambda: self.gate_off(data)),
                    'loss_delay' : tf.cond(self.control_variables['delay'],
                        lambda: self.delay_on(data), lambda: 0.0)}

                # sum losses from all train_slave_steps
                losses = {**{'loss': slave_loss['loss_gate'] +
                            slave_loss['loss_delay']},
                            **slave_loss}
                metrics = {m.name : m.result() for m in self.metrics}
                metrics.pop("loss") # remove metric duplicate
                control_states = {
                    control_name: tf.cond(
                        control_value,
                        lambda: tf.constant(True),
                        lambda: tf.constant(False),
                    )
                    for control_name, control_value in self.control_variables.items()
                }

                return {**losses, **metrics, **control_states}

        Args:
            config (Dict[str, Any]): Configuration for master step
            interpreted from default configuration file passed as the
            argumnet of the constructor.
        """
        lscope = locals()

        def _get_losses(config: Dict[str, Any]) -> str:
            """Return diconary with losses from all slave steps.
            Losses are assigned to keys representing action names from
            configuration file passed to the constructor. Each loss
            is calculated based on the value of controling variables 
            that can switch between two train_substeps or if 
            one step is missing then it will simulate train_substep
            by lambda: 0.0.

            Examples:
                # example of configuration file with two slave steps:
                1. gate - controlled by self.control_variables['gate'],
                which value is controlled callable gate_open
                2. delay - controlled by self.control_variables['delay'],
                which value is controlled callable delayopen
                
                In gate step False: {} represent step with default loss function,
                etc. set at compilation time. Because True/False attributes
                are in the "loss_gate" dictionary we bind two
                train_slave_steps: self.gate_on and self.gate_off.

                Similarely for "delay" the missing False branch for the "cond"
                represents no train_slave_step at all; thus for False evaluation
                of "delay" -> "cond" the callable is: lambda: 0.0.

                config = {"gate": { "cond": gate_open,
                    True: {"clipping": (-0.1, 0.1)}, False: {}},
                "delay": {"cond": delay_reg,
                    True: {"loss": False, "regularize": True}}}

                _get_losses(config) ->
                "{'loss_gate' : tf.cond(self.control_variables['gate'],
                lambda: self.gate_on(data), lambda: self.gate_off(data)),
                'loss_delay' : tf.cond(self.control_variables['delay'],
                lambda: self.delay_on(data), lambda: 0.0)}"

            Args:
                config (Dict[str, Any]): Configuration for master step
            interpreted from default configuration file passed as the
            argumnet of the constructor.

            Returns:
                str: string for the train_step function that
                aggregate losses from all train_slave_steps.
                It's string form is later used to bind the train_step
                funciton to the model class instance.
            """
            def _substeps_condition_execution(
                name: str, config: Dict[str, Any], on: bool
            ) -> str:
                """Helper funciton.

                Args:
                    name (str): name of the train slave taken 
                    config (Dict[str, Any]): slave step config
                    on (bool): flag indicating if we consider on/off step

                Returns:
                    str: strings for train_slave_steps callable,
                    not yet bind to the model instance.
                """
                if on:
                    return f"self.{name}_on(data)" if True in config else "0.0"
                else:
                    return f"self.{name}_off(data)" if False in config else "0.0"

            return (
                "{"
                + ",".join(
                    [
                        f"'loss_{an}' : tf.cond(self.control_variables['{an}'], lambda: {_substeps_condition_execution(an, ac, True)}, lambda: {_substeps_condition_execution(an, ac, False)})"
                        for an, ac in config.items()
                    ]
                )
                + "}"
            )

        lscope = locals()
        function_body = """
@tf.function
def train_step(self, data):
    slave_loss = {losses_config}
    # losses = {{**{{'loss': slave_loss['loss_delay'] + slave_loss['loss_gate']}}, **slave_loss}}
    losses = {{**{{'loss': slave_loss['loss_gate']}}, **slave_loss}}
    metrics = {{m.name : m.result() for m in self.metrics}}
    metrics.pop("loss")
    control_states = {{
        control_name: tf.cond(
            control_value,
            lambda: tf.constant(True),
            lambda: tf.constant(False),
        )
        for control_name, control_value in self.control_variables.items()
    }}
    
    return {{**losses, **metrics, **control_states}}
""".format(
            **{"losses_config": _get_losses(config)}
        )

        if self.verbose:
            print("\n-------------------MASTER STEP-------------------")
            print(function_body)

        """
        Execute body function within both global and local scope.
        Global scope provide libraries used inside the function body
        like: import tensroflow as tf, etc...
        Local scope provide references to the localy declared
        instance like loss funciton. 
        When we execute the funciton's body then I bind to the local
        function scope (in this case the scope of _bind_master_step).
        Then we bind the function to the instance of the model. As
        soon as the main function (in this case _bind_master_step)
        finishes execution then the funciton defined by the funcion body
        is dereferenced as well as the local scope.
        """
        exec(function_body, {**globals(), **lscope}, lscope)
        bind(self.model, lscope["train_step"])

    def _bind_slaves_steps(self, config) -> None:
        if self.verbose:
            print("-------------------SLAVE STEPS-------------------\n")

        for action_name, action_config in config.items():
            if True in action_config:
                self._bind_slave_step(action_name, action_config[True], True)
            if False in action_config:
                self._bind_slave_step(action_name, action_config[False], False)

    def _bind_slave_step(
        self, action_name: str, fn_config: Dict[str, Any], branch: bool
    ) -> None:
        lscope = {**locals(), **fn_config}
        fn_name = self._get_actoin_step_name(action_name, branch)
        if fn_config["loss"] == False:
            # dummy error that will be anyway scale by 0 to make graph to compile otherwise
            # ---> 15     retval_ = ag__.converted_call(ag__.ld(step_function), (ag__.ld(self), ag__.ld(iterator)), None, fscope)
            # ValueError: None values not supported.
            #######################################
            # OBSERVATION: loss function with y_pred and y_true must be consumed inside the gradient tape scope
            #######################################
            lscope["loss"] = self.model.compiled_loss

        function_body = f"""
@tf.function
def {fn_name}(self, data):
    x, y = data
    with tf.GradientTape(watch_accessed_variables={'False' if fn_config["variables"] or fn_config["excluded_variables"] else 'True'}) as tape:
        """

        if fn_config["variables"] or fn_config["excluded_variables"]:
            function_body += f"""
        for g in self.{'_included_variables' if fn_config["variables"] else '_excluded_variables' }['{action_name}'][{branch}]:
            tape.watch(g)
            """

        function_body += f"""
        logits = self(x, training=True)
        loss_value = {'tf.constant(0, dtype=tf.float32) *'  if fn_config["loss"]==False else ''} loss(y, logits)
        """
        # loss_value = {'loss(y, logits)' if fn_config["loss"] else 'tf.constant(0, dtype=tf.float32)'}
        # loss_value = {'loss(y, logits)' if fn_config["loss"] else 'tf.math.reduce_sum(self.losses)'}

        if fn_config["regularize"]:
            function_body += f"""
        loss_value += tf.math.reduce_sum(self.losses)
        """

        function_body += f"""
    grads = tape.gradient(loss_value, tape.watched_variables())
    self.optimizer.apply_gradients(zip({{clipping_grads}}, tape.watched_variables()))
    self.compiled_metrics.update_state(y, logits)
    return loss_value
""".format(
            **{
                "clipping_grads": "[tf.clip_by_value(g, {clip_low}, {clip_high}) for g in grads]".format(
                    **{
                        "clip_low": fn_config["clipping"][0],
                        "clip_high": fn_config["clipping"][1],
                    }
                )
                if fn_config["clipping"]
                else "grads",
                "regularize_loss_add": "loss_value += sum(self.losses)"
                if fn_config["regularize"] == True
                else "",
            }
        )

        if self.verbose:
            print(f"-------------------{fn_name}-------------------")
            print(function_body)

        exec(function_body, {**globals(), **lscope}, lscope)
        bind(self.model, lscope[fn_name])

    def on_epoch_begin(self, epoch: int, logs) -> None:
        self.epochs += 1
        self.reinit_epochs += 1
        """Control gating variable from the level of callback which can work on epoch/batch level."""
        # tf.variable.assign is different than tf.variable = <sth>. The second option is compiled to static
        # value in TF graph of computation as the result of @tf.function decorators in LoopControlableModel
        for action_name, _ in self.config.items():
            self.model.control_variables[action_name].assign(
                getattr(self, self.control_conditions[action_name])()
            )

        # early stopping
        if (
            tf.experimental.numpy.log10(
                tf.math.reduce_mean([x["loss"] for x in self.reinit_history][-3000:])
            )
            < -10
        ):
            tf.print("\nEARLY STOPPING\n")
            self.model.stop_training = True

        # reinitlaization
        if self.reinit_config and self.reinitialize_cond():
            self.reinitialize()

    def on_epoch_end(self, epoch, logs=None):
        self.model.compiled_metrics.reset_state()

    def on_batch_end(self, batch, logs):
        self.batches += 1
        self.reinit_batches += 1
        """Control gating variable from the level of callback which can work on epoch/batch level."""
        # tf.variable.assign is different than tf.variable = <sth>.
        # The second option is compiled to static value in TF graph
        # of computation as the result of @tf.function
        # decorators in LoopControlableModel
        for action_name, _ in self.config.items():
            self.model.control_variables[action_name].assign(
                getattr(self, self.control_conditions[action_name])()
            )

        self.history.append(logs)
        self.reinit_history.append(logs)

    def _get_actoin_step_name(self, action_name: str, branch: bool) -> str:
        return f"{action_name}_on" if branch else f"{action_name}_off"

    def _bind_controlers(self, config) -> None:
        self.model.control_variables = {}
        self.control_conditions = {}
        for action_name, action_config in config.items():
            self.model.control_variables[action_name] = tf.Variable(
                False, trainable=False
            )
            condition_function_name = action_name + "_condition"
            bind(self, action_config["cond"], condition_function_name)
            self.control_conditions[action_name] = condition_function_name

    def _extend_config(self, config: Dict[str, Any]) -> None:
        """Extend and validate config file. Fill missing fields based on the default_in_branch.

        Args:
            config (Dict[str, Any]): Configuration to control model training
        """

        def validate_action_config(
            action_name: str, action_config: Dict[str, Any]
        ) -> None:
            """Validate model training configuration.

            Args:
                action_name (str): name of the action slave train step
                action_config (Dict[str, Any]): configuration of the action slave train step

            Raises:
                ValueError: Missing controlable cond
                ValueError: Missing branch configuration for true/false after cond
            """

            Warning(f"------Validating Configuration for {action_name}------")
            if action_config == {}:
                Warning(
                    f"{action_name} has empty body. Condition and False or True branch must be implemented.\n It's ignored in furhter computations"
                )
            if (True not in action_config) and (False not in action_config):
                raise ValueError(
                    f"{action_name} has no False or True branch implemented"
                )
            if "cond" not in action_config:
                raise ValueError(f"{action_name} has no condition implemented.")

        # if loss in default branch is None, then use compiled loss
        if self.default_in_branch["loss"] == True:
            self.default_in_branch["loss"] = self.model.compiled_loss

        pc = {}
        for action_name, action_config in config.items():
            validate_action_config(action_name, action_config)
            pc[action_name] = {"cond": action_config["cond"]}
            if True in action_config:
                pc[action_name][True] = {
                    **self.default_in_branch,
                    **action_config[True],
                }
                if pc[action_name][True]["loss"] == True:
                    pc[action_name][True]["loss"] = self.model.compiled_loss

            if False in action_config:
                pc[action_name][False] = {
                    **self.default_in_branch,
                    **action_config[False],
                }

                if pc[action_name][False]["loss"] == True:
                    pc[action_name][False]["loss"] = self.model.compiled_loss
        return pc

    def _extract_substeps_varialbe_arrays(self, config: Dict[str, Any]) -> None:
        # keep varaibles from variable attribute from config file
        self.model._included_variables = {}
        # keep varaibles from excluded_variable attribute from config file
        self.model._excluded_variables = {}
        for action_name, action_config in config.items():
            self.model._included_variables[action_name] = {}
            self.model._excluded_variables[action_name] = {}

            if True in action_config and action_config[True]["variables"]:
                get_vars = action_config[True]["variables"]
                bind(self.model, get_vars)
                self.model._included_variables[action_name][True] = getattr(
                    self.model, get_vars.__name__
                )()
            if False in action_config and action_config[False]["variables"]:
                get_vars = action_config[False]["variables"]
                bind(self.model, get_vars)
                self.model._included_variables[action_name][False] = getattr(
                    self.model, get_vars.__name__
                )()

            if True in action_config and action_config[True]["excluded_variables"]:
                get_vars = action_config[True]["excluded_variables"]
                bind(self.model, get_vars)
                exclude_variables_names = [
                    x.name for x in getattr(self.model, get_vars.__name__)()
                ]
                self.model._excluded_variables[action_name][True] = [
                    x
                    for x in self.model.trainable_variables
                    if x.name not in exclude_variables_names
                ]

            if False in action_config and action_config[False]["excluded_variables"]:
                get_vars = action_config[False]["excluded_variables"]
                bind(self.model, get_vars)
                exclude_variables_names = [
                    x.name for x in getattr(self.model, get_vars.__name__)()
                ]
                self.model._excluded_variables[action_name][False] = [
                    x
                    for x in self.model.trainable_variables
                    if x.name not in exclude_variables_names
                ]

    def reinitialize(self):
        if self.verbose:
            tf.print("\n-------------------Reinitializatoin-------------------\n")
        self.model.reinitialize()
        self.reinit_history = []
        self.reinit_epochs = 0
        self.reinit_batches = 0

    def _bind_reinitialization(self, reinit_config: Dict[str, Any]) -> None:
        bind(self, self.reinit_config["cond"], "reinitialize_cond")

        if reinit_config["reinit_fn"]:
            Warning(
                "binding reinitalization method from reinit_config to the model instance."
            )
            bind(self.model, reinit_config["reinit_fn"], "reinitialize")
        else:
            if not callable(getattr(self.model.__class__, "reinitialize", None)):
                raise ReferenceError("model has no self.reinitialize() method")
            Warning(
                "reinit_fn from reinit_config is none: using self.reinitialize() from the model instance."
            )
