from datetime import datetime 
import uuid 
import logging
import os
import yaml
import random
from attrdict import AttrDict
import pandas as pd 
import json
import math
from pathlib import Path

def load_config(config_dir):
    with open(config_dir, 'rb') as f :
        config = yaml.load(f, Loader=yaml.FullLoader)
    args = AttrDict(config)
    return args

def generate_serial_number():
    return datetime.today().strftime('%Y-%m-%d') + '-' + str(uuid.uuid1()).split('-')[0]

def get_logger(name: str, dir_: str, stream=False) -> logging.RootLogger:

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO) # logging all levels
    
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(dir_, f'{name}.log'))

    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    if stream:
        logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    
    return logger

def load_json(data_dir, data_list):
    df = pd.DataFrame()
    for data_tuple in data_list: 
        patient_code, _ = data_tuple
        patient_id = patient_code.split('-')[0]
    
        # pft data load
        annot_file = f'{data_dir}/{patient_id}/{patient_code}/{patient_code}-annotation.json'
        with open(annot_file, 'r') as f: 
            annot_data = json.load(f)

        case_info_data = pd.DataFrame.from_dict([annot_data['Case_Info']])
        air_pollution_data = pd.DataFrame.from_dict([annot_data['Air_Pollution']])
        pft_result_data = pd.DataFrame.from_dict([annot_data['PFT_Result']])
        
        _ = pd.concat([case_info_data, air_pollution_data, pft_result_data], axis = 1)
        _ = _.set_index(keys = [[patient_code]])
        df = df.append(_)
    return df