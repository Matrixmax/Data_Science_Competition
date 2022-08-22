from cgi import test
import pandas as pd
import numpy as np
import os,gc
import matplotlib.pyplot as plt
from tqdm import *
import wandb
wandb.init(project="huaweiAI2022_CTR")

from catboost import CatBoostClassifier
from sklearn.linear_model import SGDClassifier,LinearRegression,Ridge

from sklearn.model_selection import KFold,StratifiedKFold

from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,log_loss
import warnings
warnings.filterwarnings('ignore')
os.chdir(os.getcwd()+'/huaweiAI2022_CTR')


# 读取数据
train_data_ads = pd.read_csv('./data/train/train_data_ads.csv')
train_data_feeds=pd.read_csv('data/train/train_data_feeds.csv')
test_data_ads = pd.read_csv('./data/test/test_data_ads.csv')
test_data_feeds = pd.read_csv('./data/test/test_data_feeds.csv')

# 合并数据
train_data_ads['istest'] = 0
train_data_ads['istest'] = 1
data_ads = pd.concat([train_data_ads,test_data_ads],axis=0,ignore_index=True)

train_data_feeds['istest'] = 0
test_data_feeds['istest'] = 1
data_feeds = pd.concat([train_data_feeds, test_data_feeds], axis=0, ignore_index=True)

del train_data_ads, test_data_ads, train_data_feeds, test_data_feeds
gc.collect()

# 特征工程
## 自然数编码
# 自然数编码
def label_encode(series, series2):
    unique = list(series.unique())
    return series2.map(dict(zip(
        unique, range(series.nunique())
    )))

for col in ['ad_click_list_v001','ad_click_list_v002','ad_click_list_v003','ad_close_list_v001','ad_close_list_v002','ad_close_list_v003','u_newsCatInterestsST']:
    data_ads[col] = label_encode(data_ads[col], data_ads[col])

## 特征提取

# data_feeds特征构建
cols = [f for f in data_feeds.columns if f not in ['label','istest','u_userId']]
for col in tqdm(cols):
    tmp = data_feeds.groupby(['u_userId'])[col].nunique().reset_index()
    tmp.columns = ['user_id', col+'_feeds_nuni']
    data_ads = data_ads.merge(tmp, on='user_id', how='left')

cols = [f for f in data_feeds.columns if f not in ['istest','u_userId','u_newsCatInterests','u_newsCatDislike','u_newsCatInterestsST','u_click_ca2_news','i_docId','i_s_sourceId','i_entities']]
for col in tqdm(cols):
    tmp = data_feeds.groupby(['u_userId'])[col].mean().reset_index()
    tmp.columns = ['user_id', col+'_feeds_mean']
    data_ads = data_ads.merge(tmp, on='user_id', how='left')

## 内存压缩


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
    
# 压缩使用内存
data_ads = reduce_mem_usage(data_ads)
# Mem. usage decreased to 2351.47 Mb (69.3% reduction)

# 划分训练集和测试集
cols = [f for f in data_ads.columns if f not in ['label','istest']]
x_train = data_ads[data_ads.istest==0][cols]
x_test = data_ads[data_ads.istest==1][cols]

y_train = data_ads[data_ads.istest==0]['label']

del data_ads, data_feeds
gc.collect()

# 训练模型
def cv_model(clf, train_x, train_y, test_x, clf_name, seed=2022):
    
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    train = np.zeros(train_x.shape[0])
    test = np.zeros(test_x.shape[0])

    cv_scores = []

    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} {}************************************'.format(str(i+1), str(seed)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]
               
        params = {'learning_rate': 0.3, 'depth': 5, 'l2_leaf_reg': 10, 'bootstrap_type':'Bernoulli','random_seed':seed,
                  'od_type': 'Iter', 'od_wait': 50, 'random_seed': 11, 'allow_writing_files': False}

        model = clf(iterations=20000, **params, eval_metric='AUC')
        model.fit(trn_x, trn_y, eval_set=(val_x, val_y),
                  metric_period=200,
                  cat_features=[], 
                  use_best_model=True, 
                  verbose=1)

        val_pred  = model.predict_proba(val_x)[:,1]
        test_pred = model.predict_proba(test_x)[:,1]
            
        train[valid_index] = val_pred
        test += test_pred / kf.n_splits
        cv_scores.append(roc_auc_score(val_y, val_pred))
        
        print(cv_scores)
       
    print("%s_score_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    wandb.log({'score_list:':cv_scores})

    return train, test

cat_train, cat_test = cv_model(CatBoostClassifier, x_train, y_train, x_test, "cat")

# 结果保存
x_test['pctr'] = cat_test
x_test[['log_id','pctr']].to_csv('submission.csv', index=False)