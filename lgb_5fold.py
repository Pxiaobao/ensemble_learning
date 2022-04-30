import lightgbm as lgb
import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
#这里处理完成后就变成了下面的filter2012-2021.csv
df = pd.read_csv('../merge_2012_2021_all.csv')
url1 = df[(df['snd'].notnull()) & (df['snd']!=' ') & (df['H89']!=' ') & (df['H89'].notnull())]
url1 = url1[(url1['snd']!=32766) & (url1['snd']!=0)]
data = url1[(url1['H10'] != 65534) & (url1['H10'] != 0)]
#data = pd.read_csv('filter2012-2021.csv')
xx = ['lon','lat','ele','slope','aspect','roughess','H06','V06','H07','V07','H10','V10','V18','H23','V23','V36','V89','scds']
all = ['lon','lat','ele','slope','aspect','roughess','H06','V06','H07','V07','H10','V10','V18','H23','V23','V36','V89','scds','snd']

X = data[xx]
## Y为要预测的数值
y=data["snd"]
#将数据分割训练数据与测试数据


from sklearn.model_selection import KFold
K = 5  # 折数改这里
folds = KFold(n_splits= K, shuffle = True) # 设置random_state，是为了每次都可以重复
res = []
#  这是把全部的训练数据拆分成了K份，K-1 份测试，1份验证
for trn_idx, val_idx in folds.split(X,y):
    Xtrain, ytrain = X.iloc[trn_idx,:],np.array(y)[trn_idx]
    Xtest, ytest = X.iloc[val_idx,:],np.array(y)[val_idx]
    print('1')

    train_data = lgb.Dataset(Xtrain, ytrain)
    # 将数据保存到LightGBM二进制文件将使加载更快
    test_data = lgb.Dataset(Xtest, ytest,reference=train_data) # 创建验证数据

    params = {'num_leaves': 60,#每个树上的叶子数，默认值为31，类型为int;
              'max_depth': 8,#树的最大深度 默认值：-1 取值范围：3-8（不超过10）
              'tree_learner': 'serial',
              'application': 'regression',
              'learning_rate': 0.05, #默认值：0.1,最开始可以设置得大一些，如0.1。调整完其他参数之后最后再将此参数调小。取值范围:0.01~0.3.
              'min_split_gain': 0,
            #  'min_data_in_leaf': 20,#叶子可能具有的最小记录数
              'bagging_fraction': 0.9,# 建树的样本采样比例
              'bagging_freq': 2, #bagging的次数。0表示禁用bagging，非零值表示执行k次bagging默认值：0 调参策略：3-5
              'bagging_seed': 0,
              'feature_fraction': 0.9,#例如 为0.8时，意味着在每次迭代中随机选择80％的参数来建树
              'feature_fraction_seed': 2,
              'min_sum_hessian_in_leaf': 1e-3,
              'min_child_samples':30,
              'num_threads': 1,
              'verbose': -1,
              'max_bin':100 #表示 feature 将存入的 bin 的最大数量
              }
    num_round = 500

    bst = lgb.train(params, train_data, num_round,valid_sets=test_data, early_stopping_rounds=80)

    ypred = bst.predict(Xtest, num_iteration=bst.best_iteration)

    r2score =r2_score(ytest, ypred)
    lgb_rmse = mean_squared_error(ytest, ypred) ** 0.5

    soc = 'The r2score = {}'.format(r2score)
    rmse = 'The rmse of prediction = {}'.format(lgb_rmse)
   # print(soc)
    #print(rmse)
    res.append(soc+rmse)

print(res)