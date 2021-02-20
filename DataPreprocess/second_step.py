import pandas as pd, numpy as np 
import pickle, json, ast, re, os, string
import time, datetime
import tqdm
import math, random
from collections import defaultdict
from collections import deque

from tqdm.contrib.concurrent import process_map

data_dir = '/home/data_ti5_c/fubr/CTRPrediction/CTRDatasets/'
data_dir = './Data/'
raw_data_dir = data_dir + 'raw/'
new_data_dir = data_dir + 'new/'
sample_data_dir = data_dir + 'data/'
random.seed(10)
training_data_type = 'Train'

flatten = lambda forest: [leaf for tree in forest for leaf in tree]

# #### 1. Load Data

TestSearchInfo = pd.read_csv(new_data_dir + 'TestSearchInfo.csv')
ValSearchInfo = pd.read_csv(new_data_dir + 'ValSearchInfo.csv')
TrainSearchInfo = pd.read_csv(new_data_dir + 'TrainSearchInfo.csv')

TrainSearchInfo = TrainSearchInfo.sort_values(by="SearchDate")
ValSearchInfo = ValSearchInfo.sort_values(by="SearchDate")
TestSearchInfo = TestSearchInfo.sort_values(by="SearchDate")

AdsInfo = pd.read_csv(new_data_dir + 'AdsInfo_new.csv')
AdsInfo = AdsInfo.set_index('AdID')
SearchStream = pd.read_csv(new_data_dir + 'SearchStream_new.csv')
SearchStream = SearchStream.astype("int")

data_path = raw_data_dir + 'UserInfo.tsv'
UserInfo = pd.read_csv(data_path, sep="\t")
UserInfo = UserInfo.set_index('UserID')
Category = pd.read_csv(new_data_dir + 'Category_new.csv')
Category = Category.set_index('CategoryID')
Location = pd.read_csv(new_data_dir + 'Location_new.csv')
Location = Location.set_index('LocationID')
UserInfo = UserInfo.fillna(0).astype("int")
Category = Category.fillna(0).astype("int")
Location = Location.fillna(0).astype("int")


training_SearchInfo = eval(training_data_type+'SearchInfo')
# #### 2. make search_stream_dict

SearchStream = SearchStream.sort_values(by='IsClick')
click_search_ids = SearchStream[SearchStream['IsClick']==1]['SearchID'].unique().tolist()
click_dict = dict([(y,True) for x,y in enumerate(sorted(set(click_search_ids)))])

search_stream_dict_path = new_data_dir+'search_stream_dict.pickle'
if not os.path.exists(search_stream_dict_path):
    tqdm.tqdm.pandas()
    search_stream_dict = SearchStream.groupby('SearchID')['AdID'].progress_apply(list).to_dict()
    with open(search_stream_dict_path, 'wb') as f:
         pickle.dump(search_stream_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
else:
    f = open(search_stream_dict_path, 'rb')
    search_stream_dict = pickle.load(f)


# #### 3. make user seq lists

num_ado_feature = 18
num_query_feature = 16

SearchStream = SearchStream.set_index('SearchID').sort_index()

def deal_with_user_seq(tuples_input):
    u_id = tuples_input[0]
    search_id_list = tuples_input[1]
    drop_search_id = 0
    ad_seq_list = []
    user_features = [u_id]

    new_search_id_list = []
    for search_id in search_id_list:
        if search_id not in search_stream_dict:
            drop_search_id += 1
            continue
        new_search_id_list.append(search_id)
    search_id_list = new_search_id_list

    search_id_list = list(reversed(search_id_list))
    tmp_cnt = 6
    for search_id in search_id_list:
        if tmp_cnt <= 0:
            continue
        
        if search_id in click_dict:
            tmp_cnt = 6
        ad_id_list = search_stream_dict[search_id]
        query_features = [search_id]
        search_ads_list = [query_features, ad_id_list]
        ad_seq_list.append(search_ads_list)
        tmp_cnt -= 1
        
    ad_seq_list = list(reversed(ad_seq_list))
    user_seq_list = [user_features, ad_seq_list]
    return (user_seq_list, drop_search_id)

def get_user_seq(data_type):
    drop_search_id = 0
    tmp_SearchInfo = eval(data_type+'SearchInfo')
    data_search_info_dict_path = new_data_dir+data_type+'_search_info_dict.pickle'
    if not os.path.exists(data_search_info_dict_path):
        print(tmp_SearchInfo.shape) #(95,980,006, 10) (3043334, 10) (1616503, 10)
        simple_SearchInfo = tmp_SearchInfo[['UserID', 'SearchID']]
        simple_SearchInfo_dict = simple_SearchInfo.groupby('UserID')['SearchID'].apply(list).to_dict()
        with open(data_search_info_dict_path, 'wb') as f:
            pickle.dump(simple_SearchInfo_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        f = open(data_search_info_dict_path, 'rb')
        simple_SearchInfo_dict = pickle.load(f)

    uids = tmp_SearchInfo['UserID'].unique().tolist()
    data_user_seq_lists_path = new_data_dir+data_type+'_user_seq_lists.pickle'
    if not os.path.exists(data_user_seq_lists_path):
        print(data_type, 'produce user seq')
        tmp_input = [(uid, simple_SearchInfo_dict[uid]) for uid in uids]
        Results = [deal_with_user_seq(x) for x in tmp_input]
        # Results = process_map(deal_with_user_seq, tmp_input, max_workers=num_workers, chunksize=10)
        user_seq_lists = [x[0] for x in Results] 
        drop_search_id = sum([x[1] for x in Results])
        with open(data_user_seq_lists_path, 'wb') as f:
            pickle.dump(user_seq_lists, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        f = open(data_user_seq_lists_path, 'rb')
        user_seq_lists = pickle.load(f)

    print(data_path, " drop search_ids: ", drop_search_id)
    # train: 3,438,855
    # val: 514,911 
    # test: 341,711
    return user_seq_lists
    """
    user_seq_list.pickle:
    user_seq = [uid], [ [page1], [page2], ...[pageN] ] 
    pageN = [sid], [uc_id, uc_id, uc_id, uc_id, c_id]
    """

# #### 4. produce sample

def get_query_feature(search_id, search_info):
    query_features = [search_id] + search_info[['IPID', 'IsUserLoggedOn', 'TimeStamp']].loc[search_id].astype("int").values.tolist()
    # 'LocationID', 'CategoryID'
    search_LocationID = search_info['LocationID'].loc[search_id]
    search_CategoryID = search_info['CategoryID'].loc[search_id]
    search_location_feature = [search_LocationID]+Location[['Level', 'RegionID', 'CityID']].loc[search_LocationID].values.tolist()
    search_category_feature = [search_CategoryID]+Category[['Level', 'ParentCategoryID', 'SubcategoryID']].loc[search_CategoryID].values.tolist()
    query_words = search_info['SearchQuery'].loc[search_id]
    query_words = eval(query_words)
    query_params = search_info['SearchParams'].loc[search_id]
    query_params = eval(query_params)
    query_features = query_features + search_location_feature + search_category_feature + query_words + query_params
    # 'SearchID', 'IPID', 'IsUserLoggedOn', 'TimeStamp', 'LocationID', 'CategoryID', 'Query', 'Params'
    # 4 + 4 + 4 + 1 + 3 = 16
    assert len(query_features)==num_query_feature,'query_features: '+str(len(query_features)) 
    return query_features

def get_ad_features(search_stream, ad_id):
    ad_CategoryID = AdsInfo['CategoryID'].loc[ad_id]
    ad_category_feature = [ad_CategoryID]+Category[['Level', 'ParentCategoryID', 'SubcategoryID']].loc[ad_CategoryID].values.tolist()
    ad_title = AdsInfo['Title'].loc[ad_id]
    ad_title = eval(ad_title)
    ad_params = AdsInfo['Params'].loc[ad_id]
    ad_params = eval(ad_params)

    ad_features = [ad_id] + ad_category_feature + ad_title + ad_params
    search_stream_features = search_stream[['Position', 'HistCTR', 'IsClick']].loc[ad_id].values.tolist()
    ad_features = ad_features + search_stream_features
    ## ad_id, position, ad_title_keyword, (ad_cate_id, ad_cate_level, ad_parent_cate_id, ad_sub_cate_id), hist_ctr_bin, ad_params
    # 15 + 2 + 1 = 18
    assert len(ad_features)==num_ado_feature,'ado_features: '+str(len(ad_features)) 
    return ad_features

def get_page_features(page, search_info):
    search_id = page[0][0]
    ad_id_list = page[1]

    query_features = get_query_feature(search_id, search_info)

    search_stream = SearchStream.loc[[search_id]]
    search_stream = search_stream.set_index('AdID').sort_index()

    page_ads_features = []
    cut_len = min(len(ad_id_list), 5)
    ad_id_list = ad_id_list[:cut_len]
    for ad_id in ad_id_list:
        ad_features = get_ad_features(search_stream, ad_id)
        current_ad_feature = query_features + ad_features
        # 16 + 18 = 34
        # assert len(current_ad_feature)==34,'ads_features: '+str(len(current_ad_feature)) 
        page_ads_features.append(current_ad_feature)
    for _ in range(5-len(ad_id_list)):
        page_ads_features.append(mask_ad_features)
    return flatten(page_ads_features) + [len(ad_id_list)]



sample_page_num = 5
page_ad_num = 5

num_user_features = 5
num_ad_features = num_query_feature + num_ado_feature
mask_ad_features = [0]*num_ad_features
mask_page_features = mask_ad_features*page_ad_num+[0]
num_sample_features = 1 + num_user_features + (num_ad_features-1) + page_ad_num*len(mask_page_features) + 1
num_drop_user = 0
# training_sample_list = []
# recent_histroy = defaultdict(list)


def deal_with_train_seq_list(user_seq_list):
    tmp_samples = []
    tmp_recent = defaultdict(list)
    u_id = user_seq_list[0][0]
    if u_id in UserInfo.index:
        user_features = [u_id]+UserInfo.loc[u_id].values.tolist()
    else:
        user_features = [u_id]+[0]*(num_user_features-1)
    # assert len(user_features)==5, 'user_features: '+str(len(user_features)) 
    ## UserID UserAgentID UserAgentOSID UserDeviceID UserAgentFamilyID

    page_seq = user_seq_list[1]

    search_info = training_SearchInfo.loc[[u_id]]
    search_info = search_info.set_index('SearchID').sort_index()
    
    begin_index = min(sample_page_num, len(page_seq)-1)
    # begin_index = min(5, len(page_seq))
    history_pages_deque = deque([])
    begin_pages = page_seq[:begin_index]
    # begin_pages = page_seq[-begin_index:]
    # assert len(begin_pages)<=5, "wrong length"

    for history_page in begin_pages:
        page_ads_features = get_page_features(history_page, search_info)
        history_pages_deque.append(page_ads_features)
    res_page_seq = page_seq[begin_index:]
    
    for i in range(len(res_page_seq)):
        page = res_page_seq[i]
        search_id = page[0][0]
        ad_id_list = page[1]

        search_stream = SearchStream.loc[[search_id]]
        search_stream = search_stream.set_index('AdID').sort_index()
        current_query_features = get_query_feature(search_id, search_info)

        history_pages_features = list(history_pages_deque)
        num_mask_pages = sample_page_num-len(history_pages_features)
        for _ in range(num_mask_pages):
            history_pages_features.append(mask_page_features)
        history_pages_features = flatten(history_pages_features) + [5-num_mask_pages]

        page_ads_features = []
        # cut_len = min(len(ad_id_list), 5)
        # ad_id_list = random.shuffle(ad_id_list)
        # ad_id_list = ad_id_list[:cut_len]

        is_sample_page = False
        if i == len(res_page_seq)-1:
            is_sample_page = True
        if search_id in click_dict:
            is_sample_page = True

        have_neg = False
        for ad_id in ad_id_list:
            current_ad_features = get_ad_features(search_stream, ad_id)
            current_qad_feature = current_query_features + current_ad_features
            # assert len(current_qad_feature)==33,'ads_features: '+str(len(current_ad_feature)) 
            page_ads_features.append(current_qad_feature)
            if not is_sample_page: # not click page or not last page
                continue
            if search_stream['ObjectType'].loc[ad_id] != 3: # remove nan ad
                continue
            
            label = current_ad_features[-1]
            if label !=1:
                if not have_neg and label == 0:
                    have_neg = True
                else:
                    continue

            target_ad_features = current_ad_features[:-1]
            target_qad_feature = current_query_features + target_ad_features
            target_uqad_features = user_features + target_qad_feature 
            sample = [label] + target_uqad_features + history_pages_features
            assert len(sample) == num_sample_features, "sample length wrong!!! with len: "+str(len(sample))
            # training_sample_list.append(sample)
            tmp_samples.append(sample)

        for _ in range(page_ad_num-len(ad_id_list)):
            page_ads_features.append(mask_ad_features)
        page_ads_features = flatten(page_ads_features) + [len(ad_id_list)]

        if len(history_pages_deque) >= sample_page_num:
            history_pages_deque.popleft()
        history_pages_deque.append(page_ads_features)

    history_pages_features = list(history_pages_deque)
    num_mask_pages = sample_page_num-len(history_pages_features)
    for _ in range(num_mask_pages):
        history_pages_features.append(mask_page_features)
    history_features = flatten(history_pages_features) + [sample_page_num-num_mask_pages]
    # recent_histroy[u_id] = history_features
    tmp_recent[u_id] = history_features
    return (tmp_samples, tmp_recent)

train_sample_data_path = sample_data_dir+training_data_type+'_data.csv'
if not os.path.exists(train_sample_data_path):
    print('produce train sample')
    training_user_seq_lists = get_user_seq(training_data_type)
    training_SearchInfo = training_SearchInfo.set_index('UserID').sort_index()
    new_training_user_seq_lists = []
    for user_seq_list in tqdm.tqdm(training_user_seq_lists, desc='filter train seq lists'):
        page_seq = user_seq_list[1]
        if len(page_seq) < 2:
            num_drop_user += 1
            continue
        new_training_user_seq_lists.append(user_seq_list)
    training_user_seq_lists = new_training_user_seq_lists

    # pool = Pool(10)
    num_workers = 20
    Results = process_map(deal_with_train_seq_list, training_user_seq_lists, max_workers=num_workers, chunksize=1)
    training_sample_list = []
    for x in tqdm.tqdm(Results, desc='update training sample list'):
        for y in x[0]:
            training_sample_list.append(y)
    recent_histroy = dict()
    for x in tqdm.tqdm(Results, desc='update recent history dict'):
        recent_histroy.update(x[1])

    training_sample_list = np.array(training_sample_list)
    training_data = pd.DataFrame(training_sample_list)
    print('drop_user: ', num_drop_user)
    training_data.to_csv(sample_data_dir+training_data_type+'_data.csv', index=False)

if not os.path.exists(sample_data_dir+'recent_histroy.pickle'):
    with open(sample_data_dir+'recent_history.pickle', 'wb') as f:
        pickle.dump(recent_histroy, f)
else:
    f = open(sample_data_dir+'recent_history.pickle', 'rb')
    recent_histroy = pickle.load(f)


def deal_with_validate_seq_list(user_seq_list):
    tmp_samples = []
    tmp_recent = defaultdict(list)
    u_id = user_seq_list[0][0]
    user_features = [u_id]+UserInfo.loc[u_id].values.tolist()
    search_info = validate_SearchInfo.loc[[u_id]]
    search_info = search_info.set_index('SearchID')

    recent_histroy_features = recent_histroy[u_id]
    history_features, history_page_lens = recent_histroy_features[:-1], recent_histroy_features[-1]
    history_features = np.array(history_features).reshape(-1, len(mask_page_features))
    history_features = history_features[:history_page_lens]
    history_pages_deque = deque(list(history_features))


    page_seq = user_seq_list[1]
    for i in range(len(page_seq)):
        page = page_seq[i]
        search_id = page[0][0]
        ad_id_list = page[1]
        current_query_features = get_query_feature(search_id, search_info)
        search_stream = SearchStream.loc[[search_id]]
        search_stream = search_stream.set_index('AdID').sort_index()

        history_pages_features = list(history_pages_deque)
        num_mask_pages = sample_page_num-len(history_pages_features)
        for _ in range(num_mask_pages):
            history_pages_features.append(mask_page_features)
        history_pages_features = flatten(history_pages_features) + [5-num_mask_pages]

        page_ads_features = []
        # cut_len = min(len(ad_id_list), 5)
        # ad_id_list = ad_id_list[:cut_len]


        is_sample_page = False
        if i == len(page_seq)-1:
            is_sample_page = True
        if search_id in click_dict:
            is_sample_page = True


        have_neg = False
        for ad_id in ad_id_list:
            current_ad_features = get_ad_features(search_stream, ad_id)
            current_qad_feature = current_query_features + current_ad_features
            page_ads_features.append(current_qad_feature)
            if not is_sample_page:
                continue
            if search_stream['ObjectType'].loc[ad_id] != 3:
                continue

            label = current_ad_features[-1]
            if label !=1:
                if not have_neg and label == 0:
                    have_neg = True
                else:
                    continue

            label = current_ad_features[-1]
            target_ad_features = current_ad_features[:-1]
            target_quad_features = user_features + current_query_features + target_ad_features 
            sample = [label] + target_quad_features + history_pages_features
            assert len(sample) == num_sample_features, "sample length wrong!!! with len: "+str(len(sample))
            tmp_samples.append(sample)
            # validate_sample_list.append(sample)
        
        for _ in range(page_ad_num-len(ad_id_list)):
            page_ads_features.append(mask_ad_features)
        page_ads_features =  flatten(page_ads_features) + [len(ad_id_list)]

        if len(history_pages_deque) >= sample_page_num:
            history_pages_deque.popleft()
        history_pages_deque.append(page_ads_features)

    history_pages_features = list(history_pages_deque)
    num_mask_pages = sample_page_num-len(history_pages_features)
    for _ in range(num_mask_pages):
        history_pages_features.append(mask_page_features)
    history_features = flatten(history_pages_features) + [sample_page_num-num_mask_pages]
    # recent_histroy[u_id] = history_features
    tmp_recent[u_id] = history_features
    return (tmp_samples, tmp_recent)

print('begin produce val/test sample')
for validate_data_type in ['Val', 'Test']:
    validate_SearchInfo = eval(validate_data_type+'SearchInfo')
    validate_sample_list = []
    validate_drop_user = 0

    validate_user_seq_lists = get_user_seq(validate_data_type)
    validate_SearchInfo = validate_SearchInfo.set_index('UserID').sort_index()
    new_validate_user_seq_lists = []
    for user_seq_list in tqdm.tqdm(validate_user_seq_lists, desc='filte val seq lists'):
        u_id = user_seq_list[0][0]
        if u_id not in recent_histroy.keys(): # remove cold-start user 
            validate_drop_user += 1
            continue
        new_validate_user_seq_lists.append(user_seq_list)
    validate_user_seq_lists = new_validate_user_seq_lists

    # pool = Pool(10)
    num_workers = 20
    Results = process_map(deal_with_validate_seq_list, validate_user_seq_lists, max_workers=num_workers, chunksize=1)
    validate_sample_list = []
    for x in tqdm.tqdm(Results, desc='update '+validate_data_type+' sample list'):
        for y in x[0]:
            validate_sample_list.append(y)
    recent_histroy = dict()
    for x in tqdm.tqdm(Results, desc='update recent history dict'):
        recent_histroy.update(x[1])

    validate_sample_list = np.array(validate_sample_list)
    validate_data = pd.DataFrame(validate_sample_list)
    print(validate_data_type, ' data:', validate_data.shape)
    # current
    # Val  data: (1,160,184, 895)
    # Val  drop_user:  327826
    # Test  data: (173,798, 895)
    # Test  drop_user:  311017

    # small
    # Val  data: (308,350, 895)
    # Val  drop_user:  327,826
    # Test  data: (43,174, 895)
    # Test  drop_user:  311,017

    # old
    # Val  data: (314,663, 895)
    # Test  data: (43,957, 895)
    print(validate_data_type,' drop_user: ', validate_drop_user) 
    validate_data.to_csv(sample_data_dir+validate_data_type+'_data.csv', index=False)