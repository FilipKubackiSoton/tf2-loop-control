# Loop Control Callback Dictionary of Configuration
---

## Configuration

### Available  Loop Config Atributes
```yaml
# all of the properties are accessible via method class assignemnt. In practice you can write a function pretending it's withing the class of LoopControll (e.g. using self)
history: List[tf.Float]
epoch: int
total_steps: int
steps_in_epoch: int
test_
```

### Configuration Dictionary

```yaml
action_name:
    {
        loss_function: [str, tf.loss] , # tensroflow loss funciton
        variables: [None | List[tf.Variables]] , # variables to take part in training step, if None all varaibles works
        clipping: [None | Tuple[float, float]] , # gradient clipping, if None no clipping
        exclude_variables [None | List[tf.Variables]] , # exclude variables from training step, if None no varaibles are excluded
        delay: {
            # if not specified then it's excluded from parsing
            epoch: [int] , # number of epochs when these action would not be active
            step: [int] , # number of training steps when these action would not be active
            cond: [callable] , # condition controlling delay instead of hard typed values
            join: ["and", "or"] , # if and all abovemention must be true if or only one must be valid
        }
        cond: { # condition under which 
            function: [callable] , #function defined 
        }
        # OPTIONAL
        # work_along [str]: name of the action_name' that works together when both are active
        # 
    }
```


```yaml
cond: {
    call: callable_condition_function,
    true: {
        variables:
        loss_function: 
        clipping: 
    },
    false: {
        variables:
        loss_function: 
        clipping: 
    }
}
```

```python
def delayed_regularization(self):
    """
    !!!Number of steps depends on batchsize!!!
    Impelment the function pretending it's the part of the class 
    having configuration attributes.
    """
    EPOCH_DELAY, STEPS_DELAY = 10, 1000
    return self.epoch > EPOCH_DELAY and self.total_steps > STEPS_DELAY

```

```yaml
cond: {
    call: delayed_regularization,
    true: {
        exclude_variables: [model.exclude_variables]
        loss_function: "mse"
        clipping: (0.1, 0.1)
    },
}
```




    
