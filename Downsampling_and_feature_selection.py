import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train = pd.read_csv('../input/talkingdata-data/small_train.csv')

valid = pd.DataFrame()

for i in range(6):
    valid = pd.concat([valid, pd.read_csv('../input/talkingdata-data/valid_{}.csv'.format(i))])
    print(i+1)
    
for df in [train, valid]:
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
##Downsampling function
def downsampling(df, frac):
    df0 = df[df['is_attributed'] == 0].sample(frac = frac, random_state = 0)
    df1 = df[df['is_attributed'] == 1]
    new_df = pd.concat([df0, df1], axis = 0)
    return new_df.sort_values('id')
    
private_valid = downsampling(valid[valid['hour'] != 12], frac = .5)

import gc
del valid
gc.collect()

for df in [train, private_valid]:
    for i,col in enumerate(df.columns):
        if df[col].dtype == np.int64:
            df[col] = df[col].astype(np.uint32)
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
        print(i+1)
        
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
params['learning_rate'] = 0.2
params['num_leaves'] = 32
params['max_depth'] = -1
params['max_bin'] = 300
params['min_child_samples'] = 10
params['feature_fraction'] = 0.7
params['subsample'] = 0.7
params['subsample_freq'] = 10
params['lambda_l1'] = 30
params['lambda_l2'] = 10

features = [col for col in train.columns[6:] if col != 'id' and col != 'is_attributed']
new_features = []

cat_features = ['ip', 'app', 'device', 'os', 'channel']
model = LightGBM(params, cat_features)
model.fit(train[list(train.columns[:6]) + ['is_attributed'] + new_features], 
        private_valid[list(train.columns[:6]) + ['is_attributed'] + new_features], rounds = 200, verbose = 20, early_stop = 20)

score = model.model.best_score['valid_1']['auc']


print('Base score: {}'.format(score))

import random
random.shuffle(features)

for feature in features:
    gc.collect()
    model.fit(train[list(train.columns[:6]) + ['is_attributed'] + new_features + [feature]], private_valid[list(train.columns[:6]) + ['is_attributed'] + new_features + [feature]], rounds = 200, verbose = None, early_stop = 20)
    new_score = model.model.best_score['valid_1']['auc']
    print('With feature {} model scored {}'.format(feature, new_score))
    if new_score > score:
        new_features.append(feature)
        print('Adding feature {}'.format(feature))
        score = new_score
    print('Current score: {}'.format(score))
    
print('New features: {}'.format(new_features))
    

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
    df = df[list(train.columns[:6]) + ['is_attributed'] + new_features]
    
    
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
model.fit(train, rounds = 2000, verbose = 50)

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
        df = df[list(train.columns[:6]) + ['click_id'] + new_features]
        
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
