import torch
import torch.nn as nn
import torch.nn.functional as F


from collections import OrderedDict 
from Models.utils.layer import Attention, MultiLayerPerceptron



class RACP(nn.Module):
    def __init__(self, Sampler, ModelSettings):
        super().__init__()

        # init args
        self.num_features_dict = Sampler.num_features_dict
        self.embed_dim = eval(ModelSettings['embed_dim'])
        dnn_dim_list = eval(ModelSettings['dnn_dim_list'])
        self.page_layer = ModelSettings['page_layer']
        self.remove_nan = eval(ModelSettings['remove_nan'])
        mha_head_num = eval(ModelSettings['mha_head_num'])

        # init model layer
        self._build_embedding_layer(self.num_features_dict) # build embeeding and cnt_*_fts
        self.ad_embed_dim = self.cnt_fts_dict['ad_embed'] * self.embed_dim
        self.tad_embed_dim = (self.cnt_fts_dict['ad_embed']-2) * self.embed_dim # remove is_click and page_click_num
        self.qy_embed_dim = self.cnt_fts_dict['qy_embed'] * self.embed_dim
        self.adq_embed_dim = self.ad_embed_dim + self.qy_embed_dim

        if self.page_layer == 'dynamic_page':
            alpha_input_dim = self.ad_embed_dim + self.tad_embed_dim
            alpha_dim_list = eval(ModelSettings['alpha_dim_list'])
            self.page_net = nn.ModuleDict({
                'gru': nn.GRU(self.ad_embed_dim, self.ad_embed_dim, num_layers=1),
                'target_to_adq': nn.Sequential(
                    nn.Linear(self.tad_embed_dim, self.ad_embed_dim), nn.ReLU()
                ),
                
                'din1': Attention(self.ad_embed_dim, ModelSettings),
                'alpha1': MultiLayerPerceptron(alpha_input_dim, alpha_dim_list, dropout=0, activation=nn.ReLU(), output_layer=True),

                'target_to_pq': nn.Sequential(
                    # nn.Linear(self.tad_embed_dim, self.adq_embed_dim), nn.ReLU()
                    nn.Linear(self.tad_embed_dim, self.ad_embed_dim), nn.ReLU()
                ),
                'mha2': nn.MultiheadAttention(self.adq_embed_dim, num_heads=mha_head_num),
                'din2': Attention(self.ad_embed_dim, ModelSettings),
                # 'din2': Attention(self.adq_embed_dim, ModelSettings),
                'alpha2': MultiLayerPerceptron(alpha_input_dim, alpha_dim_list, dropout=0, activation=nn.ReLU(), output_layer=True),

            })
        else:
            raise ValueError('unknow PIN page layer name: ', self.page_layer)

        self.atten_net = torch.nn.MultiheadAttention(self.ad_embed_dim, num_heads=mha_head_num)



        dnn_input_dim = self.cnt_fts_dict['user']*self.embed_dim + self.tad_embed_dim + self.ad_embed_dim
        # dnn_input_dim = self.cnt_fts_dict['user']*self.embed_dim + self.tad_embed_dim + self.ad_embed_dim + self.qy_embed_dim
        self.dnn_net = nn.ModuleDict({
            # 'dnn_input_bn': nn.BatchNorm1d(dnn_input_dim),
            'dnn':  MultiLayerPerceptron(dnn_input_dim, dnn_dim_list, dropout=0, activation=nn.PReLU(), output_layer=False)
        })

        self.logits_linear = nn.Linear(dnn_dim_list[-1], 1)

        self.init_weights()

    def _build_embedding_layer(self, num_features_dict):

        self.embedding_dict = nn.ModuleDict()
        self.cnt_fts_dict = OrderedDict()

        ### embedding uni value fts
        for key_name in ['user', 'ad', 'location', 'category']:
            num_features_list = num_features_dict[key_name]
            if key_name == 'ad':
                num_features_list = num_features_list[1:] # delete search_id

            self.embedding_dict[key_name] = nn.ModuleList(nn.Embedding(x, self.embed_dim) for x in num_features_list)
            self.cnt_fts_dict[key_name] = len(num_features_list)

        ### embedding multi value fts
        for key_name in ['ad_title', 'ad_params', 'search_query', 'search_params'] + ['page_click_num']:
            self.embedding_dict[key_name] = nn.Embedding(num_features_dict[key_name], self.embed_dim)

        self.cnt_fts_dict['multi'] = sum(num_features_dict['multi'].values())
        self.cnt_fts_dict['ad_embed'] = self.cnt_fts_dict['ad'] + self.cnt_fts_dict['location'] + \
            self.cnt_fts_dict['category']*2 + len(num_features_dict['multi'].values()) + 1 # + page_click_num(1)
        self.cnt_fts_dict['qy_embed'] = 13+1

    def __feature_embedding(self, features, embedding_name):
        num_col = features.shape[1]
        features_embed_list = []
        for col in range(num_col):
            features_embed_list.append(self.embedding_dict[embedding_name][col](features[:, col]))
        features_embed = torch.cat(features_embed_list, 1)
        return features_embed

    def __ad_embedding(self, ad_features, is_target=False):

        ad_embedding_dict = OrderedDict()
        loc_begin = 3
        ad_begin = 15
        fts_end = 33
        if is_target:
            fts_end -= 1
        begin_i = loc_begin

        for key_name in ['location', 'category']:
            tmp_features = ad_features[:, begin_i : begin_i + self.cnt_fts_dict[key_name]]
            ad_embedding_dict[key_name] = self.__feature_embedding(tmp_features, key_name)
            begin_i += self.cnt_fts_dict[key_name]

        for key_name in ['search_query', 'search_params']:
            tmp_features = ad_features[:, begin_i : begin_i + self.num_features_dict['multi'][key_name]]
            ad_embedding_dict[key_name] = nn.functional.embedding(tmp_features, self.embedding_dict[key_name].weight)
            begin_i += self.num_features_dict['multi'][key_name]

        begin_i = ad_begin + 1 #  ad_id
        key_name = 'category'
        tmp_features = ad_features[:, begin_i : begin_i+self.cnt_fts_dict[key_name]]
        ad_embedding_dict['ad_'+key_name] = self.__feature_embedding(tmp_features, key_name)
        begin_i += self.cnt_fts_dict[key_name]

        for key_name in ['ad_title', 'ad_params']:
            tmp_features = ad_features[:, begin_i : begin_i + self.num_features_dict['multi'][key_name]]
            ad_embedding_dict[key_name] = nn.functional.embedding(tmp_features, self.embedding_dict[key_name].weight)
            begin_i += self.num_features_dict['multi'][key_name]

        tmp_features = torch.cat( (ad_features[:, :loc_begin], ad_features[:, ad_begin].view(-1,1), ad_features[:, begin_i:fts_end]), 1)
        # ip, logged_on, timestamp, ad_id   , position, hist_ctr, is_click
        ad_embedding_dict['uni'] = self.__feature_embedding(tmp_features, 'ad')

        ad_features = torch.cat((
            ad_embedding_dict['uni'],   # [B, 5*D]
            ad_embedding_dict['location'],  # [B, 4*D]
            ad_embedding_dict['category'],  # [B, 4*D]
            ad_embedding_dict['search_query'].view(-1, self.embed_dim),    # [B, D]
            ad_embedding_dict['search_params'].sum(1).view(-1, self.embed_dim), # [B, D]
            ad_embedding_dict['ad_category'],  # [B, 4*D]
            ad_embedding_dict['ad_title'].sum(1).view(-1, self.embed_dim), # [B, D]
            ad_embedding_dict['ad_params'].sum(1).view(-1, self.embed_dim) # [B, D]
        ), 1)

        return ad_features  # [B, 21*D]

    def __query_embedding(self, query_features):
        query_embedding_dict = OrderedDict()
        loc_begin = 3
        fts_end = 15
        begin_i = loc_begin

        for key_name in ['location', 'category']:
            tmp_features = query_features[:, begin_i : begin_i + self.cnt_fts_dict[key_name]]
            query_embedding_dict[key_name] = self.__feature_embedding(tmp_features, key_name)
            begin_i += self.cnt_fts_dict[key_name]

        for key_name in ['search_query', 'search_params']:
            tmp_features = query_features[:, begin_i : begin_i + self.num_features_dict['multi'][key_name]]
            query_embedding_dict[key_name] = nn.functional.embedding(tmp_features, self.embedding_dict[key_name].weight)
            begin_i += self.num_features_dict['multi'][key_name]

        tmp_features = query_features[:, :loc_begin]
        # ip, logged_on, timestamp,
        query_embedding_dict['uni'] = self.__feature_embedding(tmp_features, 'ad')

        query_features = torch.cat((
            query_embedding_dict['uni'],   # [B, 5*D]
            query_embedding_dict['location'],  # [B, 4*D]
            query_embedding_dict['category'],  # [B, 4*D]
            query_embedding_dict['search_query'].view(-1, self.embed_dim),    # [B, D]
            query_embedding_dict['search_params'].sum(1).view(-1, self.embed_dim), # [B, D]
        ), 1)
        return query_features

    def _make_embedding_layer(self, features):
        ###  Embedding Layer
        embedding_layer = OrderedDict()
        batch_size = features.shape[0]
        embedding_layer['batch_size'] = batch_size

        cnt_user_fts = self.cnt_fts_dict['user'] # 5
        cnt_qad_fts = self.cnt_fts_dict['ad'] + self.cnt_fts_dict['location'] + \
            self.cnt_fts_dict['category']*2 + self.cnt_fts_dict['multi']  # 7 + 12 + 14 = 34
        cnt_qpage_fts = cnt_qad_fts*5 + 1 + 1 # + page_ad_num(1) + page_click_num(1) 
        fts_index_bias = cnt_user_fts+cnt_qad_fts - 1 # -1: target w/o is_click

        embedding_layer['user'] = self.__feature_embedding( features[:, :cnt_user_fts], 'user') # [B, 5*D]
        embedding_layer['target'] = self.__ad_embedding( features[:, cnt_user_fts:fts_index_bias], is_target=True) # [B, *D]

        page_size = 5
        ad_size = 5
        query_size = 15
        page_seq = []
        query_seq = []
        num_page_ads = []
        mask_nan_ad = torch.ones((batch_size, page_size, ad_size)).to(features.device)
        mask_click_ad = torch.ones((batch_size, page_size, ad_size)).to(features.device)
        for i in range(page_size):
            page_index_bias = fts_index_bias + i*cnt_qpage_fts
            ad_embed_seq = [
                self.__ad_embedding( features[:, page_index_bias+j*cnt_qad_fts : page_index_bias+(j+1)*cnt_qad_fts] )
                for j in range(ad_size)
            ]
            ad_embed_seq = torch.stack(ad_embed_seq, 1)  # [B, ad_size, 23*D]
            num_click = features[:, page_index_bias+ad_size*cnt_qad_fts+1].view(-1) #[B, ]
            num_click = self.embedding_dict['page_click_num'](num_click) # [B, D]
            num_click_a = num_click.unsqueeze(1).repeat(1, ad_size, 1) # [B, A, D]
            ad_embed_seq = torch.cat((ad_embed_seq, num_click_a), dim=2)
            page_seq.append(ad_embed_seq)
            query_embed = self.__query_embedding( features[:, page_index_bias : page_index_bias+query_size ] ) # [B, D]
            query_embed = torch.cat((query_embed, num_click), dim=1)
            query_seq.append(query_embed)
            num_ad = features[:, page_index_bias+ad_size*cnt_qad_fts].view(-1)
            num_page_ads.append(num_ad)

            for j in range(ad_size):
                ad_features = features[:, page_index_bias+j*cnt_qad_fts : page_index_bias+(j+1)*cnt_qad_fts]
                mask_nan_ad[:, i, j] = (ad_features[:, -1] == 2)
                mask_click_ad[:, i, j] = (ad_features[:, -1] == 1)

        embedding_layer['page_seq'] = torch.stack(page_seq, 1)  # [B, page_size, ad_size, D]
        embedding_layer['query_seq'] = torch.stack(query_seq, 1) # [B, page_size, D]
        embedding_layer['num_page_ads'] = torch.stack(num_page_ads, 1).view(batch_size, page_size) # [B, page_size, ]
        embedding_layer['num_pages'] = features[:, -1].view(-1)
        embedding_layer['mask_nan_ad'] = mask_nan_ad.bool()
        embedding_layer['mask_click_ad'] = mask_click_ad.bool()
        return embedding_layer

    def _page_layer(self, embedding_layer):
        ###  Interest Layer
        page_layer = OrderedDict()
        page_seq = embedding_layer['page_seq']  # [B, P, A, D]
        query_seq = embedding_layer['query_seq']  #[B, P, D]
        target = embedding_layer['target'] # [B, D]
        num_page_ads = embedding_layer['num_page_ads'] # [B, P]
        num_pages = embedding_layer['num_pages'] # [B]
        mask_nan_ad = embedding_layer['mask_nan_ad'] # [B, P, A]
        mask_click_ad = embedding_layer['mask_click_ad'] # [B, P, A]
        device = page_seq.device
        batch_size = page_seq.shape[0]
        page_size = page_seq.shape[1]
        ad_size = page_seq.shape[2]
        ad_masks = [torch.arange(ad_size, device=device).view(1,-1)\
                    .repeat(page_size, 1) < num_page_ad.view(-1, 1) \
                     for num_page_ad in num_page_ads]
        page_ad_masks = torch.stack(ad_masks).bool() # [B, P, A]
        page_masks = torch.arange(page_size, device=device).view(1,-1)\
                    .repeat(batch_size, 1) < num_pages.view(-1, 1)
        page_masks = page_masks.bool()

        assert page_seq.shape == (batch_size, page_size, ad_size, self.ad_embed_dim), page_seq.shape
        assert page_ad_masks.shape == (batch_size, page_size, ad_size), page_ad_masks.shape
        assert page_masks.shape == (batch_size, page_size), page_masks.shape
 
        page_seq *= page_ad_masks.float().unsqueeze(-1)  # [B, P, A, D]
        assert torch.isnan(page_seq ).any()==False, 'before mha:'+str(torch.isnan(page_seq ).any())

        if self.page_layer == 'dynamic_page':
            zero_page_ads = (num_page_ads==0) # [B, P] binary
            zero_page_ads_masks = zero_page_ads.unsqueeze(2).repeat(1, 1, ad_size)
            mha_page_ad_masks =  page_ad_masks | zero_page_ads_masks

            page_rep_list = []
            tmp_target = self.page_net['target_to_adq'](target) # [B, D]
            current_query = tmp_target
            for ii in range(page_size):
                i = page_size-1 - ii
                # calculate current page

                # mha
                # current_query = current_query.view(batch_size, 1, -1)
                # current_mha_query = current_query.permute(1, 0, 2)
                # current_page = page_seq[:, i, :, :].squeeze() # [B, A, D]
                # current_mha_input = current_page.permute(1, 0, 2) # [A, B, D]
                # current_mha_masks = ~(mha_page_ad_masks[:, i, :].squeeze()) # [B, A] 
                # current_page_rep, _ = self.page_net['mha1'](
                #     query = current_mha_query,
                #     key = current_mha_input,
                #     value = current_mha_input,
                #     key_padding_mask = current_mha_masks
                # )

                # din
                current_query = current_query #[B, D]
                current_page = page_seq[:, i, :, :].squeeze() # [B, A, D]
                current_page_ad_masks = mha_page_ad_masks[:, i, :].view(batch_size, ad_size) # [B, 1, A]
                current_page_ad_attn = self.page_net['din1'](current_query, current_page, given_mask=current_page_ad_masks)
                current_page_rep = (current_page_ad_attn.view(-1, ad_size, 1) * current_page).sum(1)

                current_page_rep = current_page_rep.view(batch_size, -1)
                page_rep_list.append(current_page_rep)

                ### update query
                # current_query = current_page_rep
                #### gru update
                gru_input = current_page_rep.view(1, batch_size, -1)
                gru_h0 = current_query.view(1, batch_size, -1)
                output, hn = self.page_net['gru'](gru_input, gru_h0)
                new_query = hn.view(batch_size, -1)

                # assign query
                current_query = new_query
            
            page_seq_rep = torch.stack(page_rep_list, 1) # [B, P, D]
            assert page_seq_rep.shape==(batch_size, page_size, self.ad_embed_dim), page_seq_rep.shape

            page_query = self.page_net['target_to_pq'](target) # [B, D]
            page_attn = self.page_net['din2'](page_query, page_seq_rep, num_pages)
            # page_input = torch.cat((page_seq_rep, query_seq), dim=2)
            # page_attn = self.page_net['din2'](page_query, page_input, num_pages)
            page_rep = (page_attn.view(-1, page_size, 1) * page_seq_rep).sum(1)
            # page_rep = (page_attn.view(-1, page_size, 1) * page_input).sum(1)

            page_rep = page_rep.view(batch_size, -1)
            # assert page_rep.shape==(batch_size, self.ad_embed_dim), page_rep.shape
        else:
            raise ValueError('unknow page layer name: ', self.page_layer)
        
        page_layer['page_rep'] = page_rep
        assert torch.isnan(page_rep ).any()==False, 'after mha:'+str(torch.isnan(page_rep ).any())
        return page_layer

    def _dnn_layer(self, embedding_layer, page_layer):
        ###  Output Layer
        dnn_layer = OrderedDict()
        mlp_iput = torch.cat([
            embedding_layer['user'], # 5*embed_dim
            embedding_layer['target'], # 22*embed_dim
            page_layer['page_rep'],
        ], 1)  # [B, 68*D]
        dnn_layer['dnn_out'] = self.dnn_net['dnn'](mlp_iput)
        return dnn_layer

    def _logits_layer(self, dnn_layer):
        return self.logits_linear(dnn_layer['dnn_out'])
    

    def forward(self, features, epoch_id=0):
        """
        click_dataset features:
            # user(5), target(31), click_ads(31*N), click_ad_num(1)
        """

        embedding_layer = self._make_embedding_layer(features)
        page_layer = self._page_layer(embedding_layer)
        dnn_layer = self._dnn_layer(embedding_layer, page_layer)
        # dnn_layer = self._dnn_layer(embedding_layer, None)
        logits = self._logits_layer(dnn_layer)
        if epoch_id >0:
            print('page_ad_attn:', page_layer['page_ad_attn'][0].data.cpu().numpy())
            print('page_attn:', page_layer['page_attn'][0].data.cpu().numpy())

        return logits.squeeze()


    def loss(self, logtis, labels):
        loss = self.Loss(logtis.squeeze(), labels.float())
        return [loss]


    def init_weights(self):
        for key_name in ['user', 'ad', 'location', 'category']:
            for e in self.embedding_dict[key_name]:
                nn.init.xavier_uniform_(e.weight)
        for key_name in ['ad_title', 'ad_params', 'search_query', 'search_params']:
            nn.init.xavier_uniform_(self.embedding_dict[key_name].weight)
