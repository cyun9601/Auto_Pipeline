import os 
import sys 
import wandb
import yaml
import importlib
from pytorch_lightning import Trainer
import pytorch_lightning as pl 
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime, timezone, timedelta
from shutil import copyfile
from modules.utils import get_logger, load_config, generate_serial_number
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

prj_dir = os.path.dirname(__file__)
os.chdir(prj_dir)
sys.path.append(prj_dir)

# train config  
CONFIG_PATH = os.path.join(prj_dir, 'config/train.yaml')

with open(CONFIG_PATH, 'rb') as f :
    args = yaml.load(f, Loader=yaml.FullLoader)
    
# with open('a.yaml', 'w') as f:
#     yaml.dump(args, f, indent = 4, allow_unicode = True)

# args = load_config(CONFIG_PATH)

# debug mode
if args['mode'] == 'debug': 
    serial_number = 'debug'
    args['model']['train']['min_epochs'] = 1
    args['model']['train']['max_epochs'] = 2
elif args['mode'] == 'train':
    # Train serial
    # serial_number = generate_serial_number()
    kst = timezone(timedelta(hours=9))
    serial_number = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")

model_name = f"{args['model']['data']['input']}-{args['model']['data']['output']}-{serial_number}"
print(f'Model Name: {model_name}')

RESULT_DIR = f'./results/train/{model_name}/'
os.makedirs(RESULT_DIR, exist_ok=True)

#-- save the config 
copyfile(CONFIG_PATH, RESULT_DIR + '/train.yaml')
pl.seed_everything(args['seed'])

#-- Logger
# Set logger
offline_logger = get_logger(name='train', dir_='./', stream=False)
offline_logger.info(f"Set Logger {RESULT_DIR}")

# callback
## checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath = RESULT_DIR + 'checkpoints/', # 체크포인트 저장 위치와 이름 형식 지정
    filename = 'best_param', # '{epoch:d}'
    verbose = True, # 체크포인트 저장 결과 출력
    save_last = True, # 마지막 체크포인트 저장
    # save_top_k = SAVE_TOP_K, # 최대 몇 개의 체크포인트를 저장할 지 지정. save_last에 의해 저장되는 체크포인트는 제외 
    monitor = f"val_{args['model']['train']['criteria']}", # 어떤 metric을 기준으로 체크포인트를 저장할 지 결정
    mode = 'min' # 지정한 metric의 어떤 기준으로 체크포인트를 저장할 지 지정
)

## early stopping callback 
early_stopping_callback = EarlyStopping(
    monitor = f"val_{args['model']['train']['criteria']}", # 모니터링할 Metric을 지정
    patience = args['model']['callbacks']['patience'],
    verbose = True, # 진행 결과 출력 
    mode = 'min' # metric을 어떤 기준으로 성능을 측정할 지 결정
)

# wandb logger
wandb_logger = WandbLogger(project = 'LG', name = f'{model_name}')

# data. train_loader, val_loader, test_loader를 반환 받아야 함.
custom_data_module = getattr(importlib.import_module('modules.dataset'), 'CustomDataModule')
dataset = custom_data_module(args['model']['data'])

model_class = getattr(importlib.import_module('model.model'), 'Model')
model = model_class(args['model'])

trainer_args = {
    'callbacks': [checkpoint_callback, early_stopping_callback],
    'gpus': args['gpus'],
    'min_epochs': args['model']['train']['min_epochs'],
    'max_epochs': args['model']['train']['max_epochs'],
    'logger': wandb_logger
}

# 학습 시작
trainer = Trainer(**trainer_args)
trainer.fit(model, dataset)

# 결과 test해보기
trainer.test(dataloaders = dataset.test_dataloader())
result = trainer.predict(dataloaders = dataset.test_dataloader())
wandb.finish()

# ##########################################RESULT_PART############################################################

print('--------------------------')
print('Using_Model')
print(f"result : {result}")
print('--------------------------')
