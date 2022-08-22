import pandas as pd
import os
os.chdir(os.getcwd()+'/huaweiAI2022_CTR')

# 只使用目标域用户行为数据
train_ads = pd.read_csv('./data/train/train_data_ads.csv',
    usecols=['log_id', 'label', 'user_id', 'age', 'gender', 'residence', 'device_name',
        'device_size', 'net_type', 'task_id', 'adv_id', 'creat_type_cd'])
test_ads = pd.read_csv('./data/test/test_data_ads.csv',
    usecols=['log_id', 'user_id', 'age', 'gender', 'residence', 'device_name',
        'device_size', 'net_type', 'task_id', 'adv_id', 'creat_type_cd'])

# 数据集采样
train_ads= pd.concat([
    train_ads[train_ads['label'] == 0].sample(70000),
    train_ads[train_ads['label'] == 1].sample(10000)
])

# 模型训练
# 加载逻辑回归
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=10000)
clf.fit(
    train_ads.drop(['log_id','label','user_id'],axis=1),
    train_ads['label']
)

# 结果输出
test_ads['pctr'] = clf.predict_proba(test_ads.drop(['log_id','user_id'],axis=1),)[:,1]
test_ads[['log_id','pctr']].to_csv('submission.csv',index=None)