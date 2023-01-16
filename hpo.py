import sys
import torch
import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from vdsr import Net
from dataset import DatasetFromHdf5
from nni.experiment import Experiment

# prune_model = torch.load('/content/gdrive/Shareddrives/High Performance ML and Real time AI/Final project/Weights/model_epoch_fpgm_0.5.pth')

search_space = {
    'weight_decay'  : {'_type': 'choice', '_value': [1e-4,1e-5]},
    'lr'            : {'_type': 'loguniform', '_value': [0.05, 1]},
    'momentum'      : {'_type': 'uniform', '_value': [0, 1]},
    'batchSize'     : {'_type': 'choice', '_value': [64, 128, 256, 512]},
    'clip'          : {'_type': 'uniform', '_value': [0, 1]}
}

experiment = Experiment('local')

experiment.config.trial_command = 'python main_vdsr_hpo.py --cuda --gpus 0'
experiment.config.trial_code_directory = '.'

experiment.config.search_space = search_space

experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'minimize'

experiment.config.max_trial_number = 10
experiment.config.trial_concurrency = 1

experiment.run(8081)
experiment.stop()