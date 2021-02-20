import numpy as np, pandas as pd
import pickle
from tqdm import tqdm
import random
import time

import torch.utils.data

class PageDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        print("Init Page features")
        dataset = np.array(dataset)
        self.labels = dataset[:, 0]

        num_ad_fts = 34
        ad_size = 5
        page_size = 5
        num_user_fts = 5
        num_page_fts = num_ad_fts*ad_size+1
        index_target_search_id = 5
        page_begin = num_user_fts + num_ad_fts - 1

        features = dataset[:, 1:]
        del dataset
        print(features.shape)
        data_len = features.shape[0]

        other_sample_features = features[:, :page_begin]
        self.target_page_id = other_sample_features[:, index_target_search_id]
        other_sample_features = np.delete(other_sample_features, index_target_search_id, axis=1) # drop target search_id

        pages = features[:, page_begin:-1].reshape(-1, num_page_fts) # only keep page_seq and drop page_num
        page_num = features[:, -1]
        del features

        # count click ad nums for each sample
        click_indexs = [num_page_fts-1-1-i*num_ad_fts for i in range(ad_size)] 
        page_click_num = sum([pages[:, click_index]==1 for click_index in click_indexs])
      

        ads = pages[:, :-1].reshape(-1, num_ad_fts) # drop ad_num
        ad_num = pages[:, -1]
        pages = ads[:, 1:].reshape(pages.shape[0], -1) # drop search_id
        del ads
        features = np.concatenate((  # add ad_num and click num
            pages,
            ad_num.reshape(-1, 1),
            np.array(page_click_num).reshape(-1,1),
        ), axis=1).reshape(data_len, -1) # [B, fts_num]
        del pages
        self.features = np.concatenate((
            other_sample_features,
            features,
            page_num.reshape(-1, 1)
        ), axis=1)
        del features
        print(self.features.shape)


    def __getitem__(self, index):
        return [self.labels[index], self.features[index], self.target_page_id[index]]

    def __len__(self):
        return len(self.labels)




