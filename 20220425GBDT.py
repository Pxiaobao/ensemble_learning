import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
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

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
K = 5  # 折数改这里
folds = KFold(n_splits= K, shuffle = True) # 设置random_state，是为了每次都可以重复
res = []
#  这是把全部的训练数据拆分成了K份，K-1 份测试，1份验证
for trn_idx, val_idx in folds.split(X,y):
    Xtrain, ytrain = X.iloc[trn_idx,:],np.array(y)[trn_idx]
    Xtest, ytest = X.iloc[val_idx,:],np.array(y)[val_idx]
    gbc = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=5)
    gbc.fit(Xtrain, ytrain)
    gbc_pred = gbc.predict(Xtest)
    gbc_score = gbc.score(Xtest, ytest)
    gbc_rmse = mean_squared_error(ytest, gbc_pred) ** 0.5

    print('The accuracy = {}'.format(gbc_score))
    print('The rmse of prodiction is:', gbc_rmse)