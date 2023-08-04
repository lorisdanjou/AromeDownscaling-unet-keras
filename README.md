# AromeDownscaling-unet-keras
This project is a part of an internship at Meteo France, which aims at downscaling meteorological fields obtained with the Arome model.
It is fully usable as long as it is used on the data it has been created to work with.

It implements a Unet, and is isnpired by https://github.com/antoinedoury/RCM-Emulator.
A second project implements a DDPM specifically for wind downscaling: https://github.com/lorisdanjou/AromeDownscaling-ddpm-pytorch

## Prerequisites
Some packages (and specific versions of these packages) need to be installed to run the codes:
```
pip install -r requirements.txt
```

## Structure of the project
The `Data` directory contains all the functions needed to load and preprocess data, `unet` contains all the necessary functions to define a Unet, a UResnet or a ResUnet, `training` contains all the functions needed to train the models (loss functions, etc.) and the `results` directory provides the necessary stuff to print results. Other useful functions are contained in `core` and `utils`.

All these functions are called by high-level scripts, defined to do a specific task.

## Define and train a model
To define and train a model, you should use the `sr.py` script. Like all other scripts in this project, it is not necessary to change the code to change a hyperparameter. All the options are specified in a configuration `.json` file. For example, you can call:

```
python3 sr.py -c config/sr_example.jsonc
```

This script can also be used to load a pretrained model and resume training, with the `weights` option in the configuration file.

## Import a trained model and make an inference
The `infer.py` file is made for that: 
```
python3 infer.py -c config/sr_example.jsonc
```

## Plot the results of an experiment
You can use the `plot_results.py` script to do that: 
```
python3 plot_results.py -c config/sr_example.jsonc
```
## Print the results of several experiments to compare them
You can use the `plot_synthesis.py` script for that task: 
```
python3 plot_synthesis.py -c config/synthesis_example.jsonc
```



