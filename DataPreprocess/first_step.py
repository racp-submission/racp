import pandas as pd, numpy as np 
import pickle, json, ast, re, os, string
import time, datetime
import tqdm
import math
from collections import defaultdict



data_dir = './Data/'
raw_data_dir = data_dir + 'raw/'
new_data_dir = data_dir + 'new/'


# ### Load Data

data_path = raw_data_dir + 'trainSearchStream.tsv'
SearchStream = pd.read_csv(data_path, sep="\t")

data_path = raw_data_dir + 'SearchInfo.tsv'
SearchInfo = pd.read_csv(data_path, sep="\t")

## SearchInfo.shape: (112,159,462, 9)
# SearchID	SearchDate	IPID	UserID	IsUserLoggedOn	SearchQuery	LocationID	CategoryID	SearchParams
## SearchInfo['UserID'].nunique(): 4,295,465
## SearchInfo['SearchID'].nunique(): 112,159,462

## SearchStream.shape (392,356,948, 6)
## SearchStream['SearchID'].nunique(): 107,863,985
## SearchStream['AdID'].nunique(): 22,848,857

data_path = raw_data_dir + 'AdsInfo.tsv'
AdsInfo = pd.read_csv(data_path, sep="\t")

data_path = raw_data_dir + 'UserInfo.tsv'
UserInfo = pd.read_csv(data_path, sep="\t")
data_path = raw_data_dir + 'Category.tsv'
Category = pd.read_csv(data_path, sep="\t")
data_path = raw_data_dir + 'Location.tsv'
Location = pd.read_csv(data_path, sep="\t")

UserInfo = UserInfo.fillna(0).astype("int")
Category = Category.fillna(0).astype("int")
Location = Location.fillna(0).astype("int")


SearchInfo = SearchInfo[['UserID', 'SearchID', 'SearchDate', 'IPID', 'IsUserLoggedOn', 'SearchQuery', 'LocationID', 'CategoryID', 'SearchParams']]
SearchStream = SearchStream[['SearchID', 'AdID', 'Position', 'ObjectType', 'HistCTR', 'IsClick']]

AdsInfo = AdsInfo[['AdID', 'Title', 'CategoryID', 'Params']]
UserInfo = UserInfo[['UserID', 'UserAgentID', 'UserAgentOSID', 'UserDeviceID', 'UserAgentFamilyID']]
Category = Category[['CategoryID', 'Level', 'ParentCategoryID', 'SubcategoryID']]
Location = Location[['LocationID', 'Level', 'RegionID', 'CityID']]

"""
features need map:
SearchID, SearchQuery, SearchParams, Titel, Params, CategoryID, LocationID

DataFrame nedd map:
SearchInfo: SearchID, SearchQuery, SearchParams, CategoryID, LocationID
SearchStream: SearchID, HistCTR
AdsInfo: 'Title', 'CategoryID', 'Params'
Category: CategoryID
Location: LocationID
"""

# #### 1. implement map function

zero_list = ['0', 0]
def preprocess_string(input):
    input = input.strip()
    exclist = string.punctuation + '\t'
    table_ = str.maketrans('', '', exclist)
    input = input.translate(table_)
    return input.strip()

def param_to_str(param, max_len_per_slot_aux):
    param_list = [  preprocess_string(x) for x in param.split(",")][:max_len_per_slot_aux]
    return '|'.join(param_list)

def words_to_str(words, max_len_per_slot_aux):
    words = preprocess_string(words)
    word_list = words.split(" ")[:max_len_per_slot_aux]
    return '|'.join(word_list)

def word_to_token(words, word_dict, sep, zero_list, max_len_per_slot_aux):
    token_list = []
    if words in zero_list:
        word_list = []
    else:
        word_list = list(filter(None, words.split(sep)))
    for i in range(max_len_per_slot_aux):
        if i < len(word_list):
            word = preprocess_string(word_list[i])
            token = word_dict[word]
            token_list.append(token)
        else:
            token_list.append(0)
    return token_list



# #### 2.  map Location: LocationID
if not os.path.exists(new_data_dir+'location_ids_dict.pickle'):
    location_ids = Location['LocationID'].unique().tolist()
    location_ids_dict = dict([(y,x+1) for x,y in enumerate(sorted(location_ids))])
    with open(new_data_dir+'location_ids_dict.pickle', 'wb') as f:
        pickle.dump(location_ids_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
else:
    f = open(new_data_dir+'location_ids_dict.pickle', 'rb')
    location_ids_dict = pickle.load(f)
tqdm.tqdm.pandas()
Location['LocationID'] = Location['LocationID'].progress_apply(lambda x: location_ids_dict[x])
Location.to_csv(new_data_dir+'Location_new.csv', index=False)
print('Finishing mapping Location')


# #### 3.  map Category: CategoryID
if not os.path.exists(new_data_dir+'category_ids_dict.pickle'):
    category_ids = Category['CategoryID'].unique().tolist()
    category_ids_dict = dict([(y,x+1) for x,y in enumerate(sorted(category_ids))])
    with open(new_data_dir+'category_ids_dict.pickle', 'wb') as f:
        pickle.dump(category_ids_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
else:
    f = open(new_data_dir+'category_ids_dict.pickle', 'rb')
    category_ids_dict = pickle.load(f)
tqdm.tqdm.pandas()
Category['CategoryID'] = Category['CategoryID'].progress_apply(lambda x: category_ids_dict[x])
Category.to_csv(new_data_dir+'Category_new.csv', index=False)
print('Finishing mapping Category')



# #### 4.  map AdsInfo: 'Title', 'CategoryID', 'Params'

# Ads params: convert Multivalent categorical strings to discrete index
ads_params = AdsInfo['Params'][AdsInfo['Params'].notnull()].tolist()
ads_params = [param_to_str(x, 5) for x in ads_params]
AdsInfo['Params'][AdsInfo['Params'].notnull()] = ads_params
if not os.path.exists(new_data_dir+'ads_params_dict.pickle'):
    ads_params = "|".join(ads_params)
    ads_params = ads_params.split("|")
    ads_params_dict = dict([(y,x+1) for x,y in enumerate(sorted(set(ads_params)))])
    with open(new_data_dir+'ads_params_dict.pickle', 'wb') as f:
        pickle.dump(ads_params_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
else:
    f = open(new_data_dir+'ads_params_dict.pickle', 'rb')
    ads_params_dict = pickle.load(f)

# Ads query: convert Multivalent categorical strings to discrete index
ads_title = AdsInfo['Title'][AdsInfo['Title'].notnull()].tolist()
ads_title = [words_to_str(x, 5) for x in ads_title]
AdsInfo['Title'][AdsInfo['Title'].notnull()] = ads_title
if not os.path.exists(new_data_dir+'ads_title_dict.pickle'):
    ads_title = "|".join(ads_title)
    ads_title = ads_title.split("|")
    ads_title_dict = dict([(y,x+1) for x,y in enumerate(sorted(set(ads_title)))])
    with open(new_data_dir+'ads_title_dict.pickle', 'wb') as f:
        pickle.dump(ads_title_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
else:
    f = open(new_data_dir+'ads_title_dict.pickle', 'rb')
    ads_title_dict = pickle.load(f)


tqdm.tqdm.pandas()
AdsInfo = AdsInfo.fillna(0)
AdsInfo['Title'] = AdsInfo['Title'].progress_apply(lambda x: word_to_token(str(x), ads_title_dict, '|', zero_list, 5))
AdsInfo['Params'] = AdsInfo['Params'].progress_apply(lambda x: word_to_token(str(x), ads_params_dict, '|', zero_list, 5))
AdsInfo['CategoryID'] = AdsInfo['CategoryID'].progress_apply(lambda x: category_ids_dict[x])
AdsInfo.to_csv(new_data_dir+'AdsInfo_new.csv', index=False)
print('Finishing mapping AdsInfo')



# #### 5.  map SearchStream: SearchID, HistCTR

num_HistCTR_type = 10
HistCTR_D = SearchStream[(SearchStream['ObjectType']==3)]['HistCTR'].tolist()
HistCTR_D = pd.qcut(HistCTR_D, num_HistCTR_type, labels=False)+1
SearchStream['HistCTR'][(SearchStream['ObjectType']==3)] = HistCTR_D
SearchStream['HistCTR'][(SearchStream['ObjectType']!=3)] = 0
SearchStream['IsClick'][(SearchStream['ObjectType']!=3)] = 2



# #### 6.  map SearchInfo: SearchID, SearchQuery, SearchParams, CategoryID, LocationID

# search params: convert Multivalent categorical strings to discrete index
search_params = SearchInfo['SearchParams'][SearchInfo['SearchParams'].notnull()].tolist()
search_params = [param_to_str(x, 3) for x in search_params]
SearchInfo['SearchParams'][SearchInfo['SearchParams'].notnull()] = search_params
if not os.path.exists(new_data_dir+'search_params_dict.pickle'):
    search_params = "|".join(search_params)
    search_params = search_params.split("|")
    search_params_dict = dict([(y,x+1) for x,y in enumerate(sorted(set(search_params)))])
    with open(new_data_dir+'search_params_dict.pickle', 'wb') as f:
        pickle.dump(search_params_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
else:
    f = open(new_data_dir+'search_params_dict.pickle', 'rb')
    search_params_dict = pickle.load(f)


# search query: convert Multivalent categorical strings to discrete index
search_querys = SearchInfo['SearchQuery'][SearchInfo['SearchQuery'].notnull()].tolist()
search_querys = [words_to_str(x, 1) for x in search_querys]
SearchInfo['SearchQuery'][SearchInfo['SearchQuery'].notnull()] = search_querys
if not os.path.exists(new_data_dir+'search_querys_dict.pickle'):
    search_querys = "|".join(search_querys)
    search_querys = search_querys.split("|")
    search_querys_dict = dict([(y,x+1) for x,y in enumerate(sorted(set(search_querys)))])
    with open(new_data_dir+'search_querys_dict.pickle', 'wb') as f:
        pickle.dump(search_querys_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
else:
    f = open(new_data_dir+'search_querys_dict.pickle', 'rb')
    search_querys_dict = pickle.load(f)

## search ids
if not os.path.exists(new_data_dir+'search_ids_dict.pickle'):
    search_ids = set(SearchInfo['SearchID'].unique().tolist() + SearchStream['SearchID'].unique().tolist())
    search_ids = list(search_ids)
    search_ids_dict = dict([(y,x+1) for x,y in enumerate(sorted(search_ids))])
    with open(new_data_dir+'search_ids_dict.pickle', 'wb') as f:
        pickle.dump(search_ids_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
else:
    f = open(new_data_dir+'search_ids_dict.pickle', 'rb')
    search_ids_dict = pickle.load(f)


# covert 
tqdm.tqdm.pandas()
SearchStream['SearchID'] = SearchStream['SearchID'].progress_apply(lambda x: search_ids_dict[x])
SearchStream.to_csv(new_data_dir+'SearchStream_new.csv', index=False)

SearchInfo = SearchInfo.fillna(0)
SearchInfo['SearchID'] = SearchInfo['SearchID'].progress_apply(lambda x: search_ids_dict[x])
SearchInfo['CategoryID'] = SearchInfo['CategoryID'].progress_apply(lambda x: category_ids_dict[x])
SearchInfo['LocationID'] = SearchInfo['LocationID'].progress_apply(lambda x: location_ids_dict[x])
SearchInfo['SearchQuery'] = SearchInfo['SearchQuery'].progress_apply(lambda x: word_to_token(str(x), search_querys_dict, '|', zero_list, 1))
SearchInfo['SearchParams'] = SearchInfo['SearchParams'].progress_apply(lambda x: word_to_token(str(x), search_params_dict, '|', zero_list, 3))
SearchInfo['SearchDate'] = pd.to_datetime(SearchInfo['SearchDate'])
SearchInfo.SearchDate = SearchInfo.SearchDate.values.astype(np.int64) // 10 ** 9
num_Date_type = 300
Date_D = SearchInfo['SearchDate'].tolist()
Date_D = pd.qcut(Date_D, num_Date_type, labels=False)+1
SearchInfo['TimeStamp'] = Date_D
print('Finish dealing with SearchInfo')


# #### 7. split SearchInfo by datetime

print('begin split SearchInfo')
train_begin_datetime = "2015-04-28 00:00:00"
val_begin_datetime = "2015-05-19 00:00:00"
test_begin_datetime = "2015-05-20 00:00:00"
train_begin_timestamp = time.mktime(datetime.datetime.strptime(train_begin_datetime, "%Y-%m-%d %H:%M:%S").timetuple())
val_begin_timestamp = time.mktime(datetime.datetime.strptime(val_begin_datetime, "%Y-%m-%d %H:%M:%S").timetuple())
test_begin_timestamp = time.mktime(datetime.datetime.strptime(test_begin_datetime, "%Y-%m-%d %H:%M:%S").timetuple())
val_begin_timestamp = int(val_begin_timestamp)
test_begin_timestamp = int(test_begin_timestamp)

TrainSearchInfo = SearchInfo[ ~(SearchInfo['SearchDate']<train_begin_timestamp) & (SearchInfo['SearchDate']<val_begin_timestamp) ]
ValSearchInfo = SearchInfo[ ~(SearchInfo['SearchDate']<val_begin_timestamp) & (SearchInfo['SearchDate']<test_begin_timestamp) ]
TestSearchInfo = SearchInfo[ ~(SearchInfo['SearchDate']<test_begin_timestamp) ]
print(TrainSearchInfo.shape)
print(ValSearchInfo.shape)
print(TestSearchInfo.shape)
# (972924, 9)
# (37107, 9)
# (32379, 9)
# (95980006, 9)
# (3043334, 9)
# (1616503, 9)


TrainSearchInfo.to_csv(new_data_dir+'TrainSearchInfo.csv', index=False)
ValSearchInfo.to_csv(new_data_dir+'ValSearchInfo.csv', index=False)
TestSearchInfo.to_csv(new_data_dir+'TestSearchInfo.csv', index=False)

print('finish split SearchInfo')

print('\nfinish all\n')