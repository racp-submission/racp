import os
from pandas.io.parsers import ParserBase
import tqdm
import time
import math
import random
import pickle
import numpy as np, pandas as pd
import scipy.sparse as sp
from numpy.random import choice
from copy import deepcopy
from collections import defaultdict
from collections import OrderedDict 

import torch
from torch.utils.data import DataLoader

from DatasetsLoad.dataset import *


class SampleGenerator(object):

    def __init__(self, DataSettings, TrainSettings, mode):
        self.mode = mode
        # data 
        self.DataSettings = DataSettings
        
        # data path
        data_name = DataSettings['data_name']
        self.data_dir = DataSettings['data_dir']
        self.dataset_type = DataSettings['dataset_type']

        
        # read data
        start_time = time.time()
        print("=========== reading ", data_name, " data ===========")
        print(self.data_dir)
        self._get_main_data(TrainSettings)  
        self._init_data_statistic()  

        print(f'read data file cost {time.time()-start_time} seconds\n') 
    

    def _get_main_data(self, TrainSettings):
        dataset = eval(self.dataset_type+'Dataset')

        self.Train_data, self.Val_data, self.Test_data = None, None, None
        if self.dataset_type in ['Page']:
            if self.mode == 'train':
                self.Train_data = dataset(pd.read_csv(self.data_dir+'Train_data.csv'))
                self.Val_data = dataset(pd.read_csv(self.data_dir+'Val_data.csv'))
            self.Test_data = dataset(pd.read_csv(self.data_dir+'Test_data.csv'))
        else:
            raise ValueError('unknow dataset type: ', self.dataset_type)
        
        if self.mode == 'train':
            print('Train  size: ', len(self.Train_data))
            print('Val  size: ', len(self.Val_data))
        print('Test  size: ', len(self.Test_data))
        
        # label(1), user(5), target(32), page1(166)*5, page_num(1)

        data_loader_workers = eval(TrainSettings['data_loader_workers'])
        batch_size_setting = {'Train':'train', 'Val':'test', 'Test':'test'}
        
        if self.mode == 'train':
            self.Train_data, self.Val_data, self.Test_data = [
                DataLoader(
                    getattr(self, data_type+'_data'), 
                    batch_size = eval(TrainSettings[batch_size_setting[data_type] + '_batch_size']), 
                    shuffle=True, 
                    num_workers=data_loader_workers
                )
                for data_type in ['Train', 'Val', 'Test'] 
            ]
        else:
            data_type = 'Test'
            self.Test_data = DataLoader(
                    getattr(self, data_type+'_data'), 
                    batch_size = eval(TrainSettings[batch_size_setting[data_type] + '_batch_size']), 
                    shuffle=True, 
                    num_workers=data_loader_workers
                )


    def _init_data_statistic(self):
        num_features_dict = OrderedDict()
        plus_one = lambda o: [ x+1 for x in o ] 

        num_features_dict['user'] = plus_one([4339861, 64017, 51, 4213, 89])
        # (user_id, user_agent_id, user_agent_os_id, user_device_id, user_agent_family_id)
        
        num_features_dict['ad'] = plus_one([1, 2266704, 1, 300] + [36893298, 8, 10, 2])
        # search_id, ip_id, is_user_logged_on, timestamp + ad_id, position, hist_ctr_bin, is_click

        num_features_dict['page_click_num'] = 5 + 1
        
        num_features_dict['location'] = plus_one([4080, 3, 84, 3723])
        # LocationID, Level, RegionID, CityID

        num_features_dict['category'] = plus_one([68, 3, 12, 57])
        # CategoryID, Level, ParentCategoryID, SubcategoryID

        num_features_dict['ad_title'] = 2943212 + 1
        num_features_dict['ad_params'] = 5712 + 1
        num_features_dict['search_query'] = 33732 + 1
        num_features_dict['search_params'] = 1192 + 1

        num_multi_fts = OrderedDict()
        num_multi_fts['search_query'] = 1
        num_multi_fts['search_params'] = 3
        num_multi_fts['ad_title'] = 5
        num_multi_fts['ad_params'] = 5
        num_features_dict['multi'] = num_multi_fts

        self.num_features_dict = num_features_dict
    
    def _sample_negative(self, total_set, pos_set):
        j = random.choice(total_set)
        while(j in pos_set):
            j = random.choice(total_set)
        return j