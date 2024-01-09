import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train = pd.read_csv('../input/talkingdata-data/small_train2.csv')

for df in [train]:
    df.drop(['day', 'UTC_day', 'click_time'], axis = 1, inplace = True)
    for col in df.columns:
        if col[-7:] == '_cumcnt':
            col_name = col[:-6]
            df[col_name + 'invcumcnt'] = df[col_name + 'cnt'] - df[col]
        if col[-11:] == '_next_click' or col[-11:] == '_prev_click':
            col_name = col + '_'
            df[col_name + 'small'] = (df[col] <= 3)
            df[col_name + 'weird'] = (df[col] > 0) & ((df[col] % 100 == 0) | (df[col] % 60 == 0))
    df['ip2'] = df['ip']
    df['app2'] = df['app']
    df['channel2'] = df['channel']
    
import lightgbm as lgb

class LightGBM: 
    def __init__(self, params, cat_features):
        self.params = params
        self.params['application'] = 'binary'
        self.params['metric'] = 'auc'
        self.params['nthread'] = 4
        self.cat_features = cat_features
        self.model = None
    
    def fit(self, train, valid = None, rounds = 100, verbose = 5, early_stop = None):
        def get_lgb_dataset(df):
            return lgb.Dataset(df.drop(['id', 'is_attributed'], axis = 1), df['is_attributed'])
        
        l_train = get_lgb_dataset(train)
        valid_sets = [l_train]
        
        if valid is not None:
            if isinstance(valid, list) == False: valid = [valid]
                
            for valid_set in valid:
                l_valid = get_lgb_dataset(valid_set)
                valid_sets.append(l_valid)
        
        self.model = lgb.train(self.params, l_train, valid_sets = valid_sets, num_boost_round = rounds,
                            verbose_eval = verbose, categorical_feature = self.cat_features, early_stopping_rounds = early_stop)
        
    def predict(self, df):
        assert self.model is not None, "You have to fit the model first!"
        if np.isin('click_id', df.columns): y_pred = self.model.predict(df.drop(['click_id', 'id'], axis = 1))
        else: y_pred = self.model.predict(df.drop(['id', 'is_attributed'], axis = 1))
        return y_pred
        
params = {}
params['learning_rate'] = 0.02
params['num_leaves'] = 32
params['max_depth'] = -1
params['max_bin'] = 300
params['min_child_samples'] = 10
params['feature_fraction'] = 0.7
params['subsample'] = 0.7
params['subsample_freq'] = 10
params['lambda_l1'] = 30
params['lambda_l2'] = 10

cat_features = ['ip', 'app', 'device', 'os', 'channel']
model = LightGBM(params, cat_features)

model.fit(train, rounds = 1300, verbose = 50)

import gc
click_ids = []
y_pred = []

for i in range(22):
    chunk = pd.read_csv('../input/talkingdata-data/test_{}.csv.gzip'.format(i), compression = 'gzip')
    for df in [chunk]:
        df.drop(['day', 'UTC_day'], axis = 1, inplace = True)
        for col in df.columns:
            if col[-7:] == '_cumcnt':
                col_name = col[:-6]
                df[col_name + 'invcumcnt'] = df[col_name + 'cnt'] - df[col]
            if col[-11:] == '_next_click' or col[-11:] == '_prev_click':
                col_name = col + '_'
                df[col_name + 'small'] = (df[col] <= 3)
                df[col_name + 'weird'] = (df[col] > 0) & ((df[col] % 100 == 0) | (df[col] % 60 == 0))
        df['ip2'] = df['ip']
        df['app2'] = df['app']
        df['channel2'] = df['channel']
        
    y_pred.append(model.predict(chunk))
    click_ids.append(chunk['click_id'].values)
    
    del chunk
    gc.collect()
    
    print(i+1)
    
click_ids = np.concatenate(click_ids)
y_pred = np.concatenate(y_pred)

sub = pd.DataFrame()
sub['click_id'] = click_ids
sub['is_attributed'] = y_pred
sub.sort_values('click_id', inplace = True)

assert np.all(sub['click_id'].values == np.arange(sub.shape[0]))

sub.to_csv('sub.csv', index = False, float_format = '%.8f')