import tensorflow as tf

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
    def __init__(self, config: int, *args, **kwargs) -> None:
        super(LoopControlerCallback, self).__init__(*args, **kwargs)
        self.config = config

    def on_train_begin(self, logs=None):

        # meta attributes for building conditions
        self.epochs = 0
        self.batches = 0
        self.batches_in_epoch = 0
        self.last_loss = 0.0
        self.history = []

        @tf.function
        def train_step(self, data):
            train_metrics = tf.cond(
                self.gate, lambda: self.gate_on(data), lambda: self.gate_off(data)
            )

            final_results = {
                **train_metrics,
                **{
                    c_name: tf.cond(
                        getattr(self, c_name),
                        lambda: tf.constant(True),
                        lambda: tf.constant(False),
                    )
                    for c_name in self.cv_names
                },
            }

            return final_results

        @tf.function
        def gate_on(self, data):
            x, y = data
            with tf.GradientTape(watch_accessed_variables=True) as tape:
                logits = self(x, training=True)
                loss_value = self.compiled_loss(y, logits)
            grads = tape.gradient(loss_value, tape.watched_variables())
            self.optimizer.apply_gradients(zip(grads, tape.watched_variables()))
            return {m.name: m.result() for m in self.metrics}
            

        @tf.function
        def gate_off(self, data):
            x, y = data
            with tf.GradientTape(watch_accessed_variables=True) as tape:
                logits = self(x, training=True)
                loss_value = self.compiled_loss(y, logits)
            grads = tape.gradient(loss_value, tape.watched_variables())
            self.optimizer.apply_gradients(zip(grads, tape.watched_variables()))
            return {m.name: m.result() for m in self.metrics}

            
        bind(self.model, train_step)
        bind(self.model, gate_on)
        bind(self.model, gate_off)

        self._init_cv()
        self._bind_cv()

    def _get_cv_name(self, action_name):
        return f"{action_name}_variable_control"
    
    def _get_cc_name(self, condition_name):
        return f"{condition_name}_condition_control"

    def _init_cv(self):
        self.model.cv_names = []
        for control_variable_name in self.config.keys():
            setattr(
                self.model, control_variable_name, tf.Variable(False, trainable=False)
            )
            self.model.cv_names.append(control_variable_name)

    def _bind_cv(self):
        self.c_conds = {}
        for action_name, action_config in self.config.items():
            name = self._get_cv_name(action_name)
            bind(self, action_config["cond"], name)
            self.c_conds[action_name] = name
    



    @tf.function
    def train_step_adjustable(self, data, loss):
        x, y = data
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            logits = self(x, training=True)
            loss_value = loss(y, logits)
        grads = tape.gradient(loss_value, tape.watched_variables())
        self.optimizer.apply_gradients(zip(grads, tape.watched_variables()))
        return {m.name: m.result() for m in self.metrics}

    def set_steps_functions(self, data, loss):
        for action_name, action_config in self.config.items():
            bind(self.model, action)

    
    def on_epoch_end(self, epoch, logs):
        self.epochs += 1
        """Control gating variable from the level of callback which can work on epoch/batch level."""
        # tf.variable.assign is different than tf.variable = <sth>. The second option is compiled to static
        # value in TF graph of computation as the result of @tf.function decorators in LoopControlableModel
        for control_name, control_function_name in self.c_conds.items():
            getattr(self.model, control_name).assign(
                getattr(self, control_function_name)()
            )

    def on_batch_end(self, batch, logs=None):
        self.batches += 1
        
