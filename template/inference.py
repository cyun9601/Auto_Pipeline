import os 
import sys 
import importlib
import pytorch_lightning as pl 
from shutil import copyfile
from modules.utils import get_logger, load_config
import pandas as pd
from modules.parser import Parser
import pickle
import torch
from torch.nn import functional as F

import warnings
warnings.filterwarnings("ignore")

prj_dir = os.path.dirname(__file__)
os.chdir(prj_dir)
sys.path.append(prj_dir)

# inference config  
CONFIG_PATH = os.path.join(prj_dir, 'config/inference.yaml')
args = load_config(CONFIG_PATH)

# debug mode 
print(f'Model Name: {args.model_name}')

RESULT_DIR = f'./results/inference/{args.model_name}/'
os.makedirs(RESULT_DIR, exist_ok=True)

# load train config sls
TRAIN_DIR = f'./results/train/{args.model_name}/'
TRAIN_CONFIG_PATH = os.path.join(TRAIN_DIR, 'train.yaml')
train_args = load_config(TRAIN_CONFIG_PATH)

#-- save the config 
copyfile(TRAIN_CONFIG_PATH, RESULT_DIR + '/train.yaml')
copyfile(CONFIG_PATH, RESULT_DIR + '/inference.yaml')

# Load model 
model_class = getattr(importlib.import_module('model.model'), 'Model')
model = model_class.load_from_checkpoint(f'./results/train/{args.model_name}/checkpoints/best_param.ckpt', model_args = train_args.model)
model.eval()

# Load data
### Write your code here

############################## 
pred = model(input_tensor)
l1 = F.l1_loss(pred, output_tensor, reduction = 'none').sum(axis = 1)

pred_df = pd.DataFrame(pred.detach().numpy(), columns = [f'feature_{i}' for i in range(pred.shape[1])], index = input_df.index)
loss_df = pd.DataFrame(l1.detach().numpy(), columns = ['loss'], index = input_df.index)

result = pd.concat([input_df, pred_df, loss_df], axis = 1).sort_values(f'loss', ascending=False)
result.to_csv(os.path.join(RESULT_DIR, 'result.csv'), index = False)