import pandas as pd 
# from datatype import *
import os
import time 
import re 
import numpy as np
import sys 
from tqdm.auto import tqdm
from typing import Union



signal_prefix_dict = {'vacuum':'vacuum', 'charging':'meas_pres'}

class Parser():
    def __init__(self, df, align:str, expansion:bool, samples:int = -1):
        self.df = df
        
        if samples != -1: 
            self.df = self.df.iloc[0:samples]
        
        self._drop_gentime()
        self._drop_duplicate()
        self._process_signal(signal_prefix_dict, align, expansion)
    
    @property
    def get_barcode(self):    
        return self.df.barcode.tolist()
    
    def _drop_gentime(self):
        # self.df['gen_dt'] = self.df['gen_dt'].astype('datetime64[ns]')
        # self.df['create_time'] = self.df['create_time'].astype('datetime64[ns]')
        self.df = self.df[self.df['gen_dt'] < self.df['create_time']].reset_index(drop=True)
    
    def _drop_duplicate(self):
        self.df = self.df.drop_duplicates(subset = ['barcode'], keep=False).reset_index(drop=True)

    def _process_signal(self, signal_prefix_dict:dict, align:str, expansion:bool) -> None:
        for col_prefix in tqdm(signal_prefix_dict.values()):
            regex = re.compile(rf'{col_prefix}+[0-9]')
            column_list = [col for col in self.df.columns if regex.search(col) != None]
            self._align_expansion(column_list, align, expansion)

    def get_signal(self, signal_type:Union[str, list], barcode:Union[str, list]=None, transpose:bool=False, reset_index:bool=False) -> pd.DataFrame:
        
        if isinstance(signal_type, str):
            signal_type = [signal_type]
    
        signal_col_list = ['barcode']
        for st in signal_type:  
            col_prefix = signal_prefix_dict[st]
            regex = re.compile(rf'{col_prefix}+[0-9]')
            n_signal = len([True for col in self.df.columns if regex.search(col) != None])
            signal_col_list = signal_col_list + [f'{col_prefix}{i}' for i in range(1, n_signal + 1)]
        
        if barcode == None : 
            view = self.df[signal_col_list]
        else :
            if isinstance(barcode, str): 
                barcode = [barcode]
                view = self.df.loc[self.df.barcode.isin(barcode), signal_col_list]

        # transpose 
        if transpose:
            view = view.set_index('barcode').transpose()
            return view 
        else : 
            if reset_index:
                return view
            else : 
                return view.set_index('barcode')
    
    def _align_expansion(self, column_list:list, align = 'left', expansion=True) -> None:
        _df = self.df[column_list]
        _df_colname_list = list(_df.columns)
        _df_array = _df.values
        n_data = len(_df_array)
        n_signal = len(_df.columns)
          
        for i in range(n_data):
            sensor_data = _df_array[i, :]
            sensor_data = sensor_data[~np.isnan(sensor_data)]
            
            if expansion: 
                sensor_data = self._expansion_signal(sensor_data, n_signal)
            else : 
                if align == 'left':
                    sensor_data = np.append(sensor_data, np.repeat(np.nan, n_signal - len(sensor_data))) 
                elif align == 'right':
                    sensor_data = np.append(np.repeat(np.nan, n_signal - len(sensor_data)), sensor_data) 
                else: 
                    raise "align must be 'left' or 'right'"
                        
            _df_array[i, :] = sensor_data
        self.df.loc[:, _df_colname_list] = _df_array
    
    @staticmethod
    def _expansion_signal(signal:np.array, to_length:int) -> np.array:
        xp = list(range(len(signal)))
        
        to_pred_x = np.linspace(start = 1, stop = len(signal), num=to_length)
        return np.interp(to_pred_x, xp, signal)
    
        
if __name__=="__main__":
    project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    
    vacuum = pd.read_csv(os.path.join(project_base_dir, 'data/vacuum.csv'))
    charging = pd.read_csv(os.path.join(project_base_dir, 'data/charging.csv'))
    
    merged_df = pd.merge(charging.rename(columns = {'set_id':'barcode'}), vacuum, on='barcode')
    
    parser = Parser(merged_df, align = 'left', expansion=True, samples=100)
    parser.get_signal(signal_type = ['vacuum'])