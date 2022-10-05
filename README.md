<p align="center">
    <img src="wingman.jpg" width="400"/>
</p>

# Wingman

Wingman is a Python library built for managing projects in deep learning.
Currently, Wingman is capable of automatically handling:
- Automatic weights saving and versioning
- Smooth WanbB integration
- Building blocks for popular torch modules
- Smoothly handling commandline arguments and `*.yaml` config files

## Installation

`pip3 install jj-wingman`

## Philosophy

Wingman is designed to be very horizontally integrated into your deep learning projects.
Instead of wrapping your models with wrappers or designing your models around Wingman, Wingman adapts to your projects, and you are the decider on when and where Wingman comes into play.

## Modules

Wingman has two core modules and several static functions, all of which can be used independently or in conjunction.


### `from wingman import Wingman`

This is the core module for Wingman, Wingman requires a bare minimum `settings.yaml` file somewhere accessible in your project directory.
A bare minimum yaml file is as follows:

```yaml
# wingman required params
debug: false

weights_directory: 'weights'
version_number: null
mark_number: 0

increment: true
epoch_interval: -1
batch_interval: 10
max_skips: 5
greater_than: 0.0

wandb: false
wandb_name: ''
wandb_notes: ''
wandb_id: ''
wandb_entity: 'jjshoots'
wandb_project: 'my_new_project'
```

The parameters described are as follows:
- `debug`: Whether to set the model to debug mode
- `weights_directory`: Where should Wingman point to for model weight saving
- `version_number`: Wingman versions models different models using this number, if this is left as null, Wingman automatically chooses a number.
- `mark_number`: Wingman versions different model checkpoints using this number.
- `increment`: Whether to increment mark number, if this is set to false, Wingman won't save multiple variations of the same model.
- `epoch_interval` and `batch_interval`: Training for deep learning models is generally done using epochs and batches. You can ask wingman to record training every n batches or every n epochs. Note that only one of these can be set, and the other must be -1.
- `max_skips`: How many epochs/batches (specified using the previous arguments) before Wingman will save an intermediary checkpoint of the model.
- `greater_than`: You can tell Wingman to only checkpoint model when the previous checkpointed loss is more than the current loss by this value.
- `wandb`: Whether to log things to WandB.
- `wandb_name`: The name of the run to be displayed in WandB. If left blank, Wingman automatically assigns one depending on the model version number. If left blank, one is automatically assigned.
- `wandb_notes`: Some helpful notes that you can leave for your runs that will be recorded in WandB.
- `wandb_id`: A Unique ID for this run. If left blank, one is automatically assigned.
- `wandb_entity`: Your username/organization name for WandB.
- `wandb_project`: The project that this run will be under.

If you need more parameters, adding them to the `settings.yaml` file can be done freely, and these parameters can be easily referenced in the code later.
In addition, Wingman also automatically generates a `device` parameter, this parameter is automatically set to the GPU if your system has one, and can be used later for easy transfer of tensors.

You can also use the `settings.yaml` file to define a set of sane defaults, and then override them using commandline arguments later.
This is because Wingman automatically converts all parameters in the `settings.yaml` file into commandline arguments using the `argparse` python module.
This allows your `main.py` file to be much cleaner, without requiring the 50 odd lines of code at the top only for parameter definition.

After defining your `settings.yaml` file, the basic usage of Wingman is as follows:

```python
    from wingman import Wingman

    # initialize Wingman and get all the parameters from the `settings.yaml` file
    helper = Wingman("./settings.yaml")
    set = helper.set

    # load the model and optimizer, `set.device` is automatically generated
    model = Model(set.YOUR_PARAM, set.YOUR_OTHER_PARAM).to(set.device)
    optim = optimizer.AdamW(model.parameters(), lr=set.YOUR_OTHER_PARAM, amsgrad=True)

    # we can check if we have trained this model before, and if we have, just load it
    # this checking is done using the `version_number` param, if `latest=True` is set,
    # Wingman automatically searches for the latest model checkpoint
    have_file, weight_file, optim_file = self.get_weight_files(latest=True)
    if have_file:
        # wingman simply returns a string of where the weight files are
        # no unnecessary wrapping!
        model.load(model_file)
        optim.load(optim_file)

    # let's run some training:
    while(training):
        ...
        # training code here
        loss = YOUR_LOSS_FUNCTION(model)
        ...

        # let Wingman handle checkpointing for you
        update_weights, model_file, optim_file = self.checkpoint(loss, batch_number, epoch_number)
        if update_weights:
            # if Wingman deems that the weights should be checkpointed, it returns
            # a string of where the weight files should go for it to be found later
            model.save(model_file)
            optim.save(optim_file)
```
