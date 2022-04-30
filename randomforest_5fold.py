import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
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
    # criterion ：
    # 回归树衡量分枝质量的指标，支持的标准有三种：
    # 1）输入"mse"使用均方误差mean squared error(MSE)，父节点和叶子节点之间的均方误差的差额将被用来作为特征选择的标准，
    # 这种方法通过使用叶子节点的均值来最小化L2损失
    # 2）输入“friedman_mse”使用费尔德曼均方误差，这种指标使用弗里德曼针对潜在分枝中的问题改进后的均方误差
    # 3）输入"mae"使用绝对平均误差MAE（mean absolute error），这种指标使用叶节点的中值来最小化L1损失
    forest = RandomForestRegressor(n_estimators=500,
                                   criterion='mse',
                                   min_samples_split=2,
                                   min_samples_leaf=1,
                                    random_state=1,
                                   max_depth=12,
                                   n_jobs=-1)
    forest.fit(Xtrain, ytrain)
    y_train_pred = forest.predict(Xtrain)
    y_test_pred = forest.predict(Xtest)

    soc = 'The r2score = {}'.format(r2_score(ytest, y_test_pred))
    rmse = 'The rmse of prediction = {}'.format(mean_squared_error(ytest, y_test_pred) ** 0.5)
    res.append(soc+rmse)

print(res)







