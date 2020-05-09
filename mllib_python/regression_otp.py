import pandas as pd;
import numpy as np;
import xgboost as xgb;
import itertools;
from sklearn.preprocessing import LabelEncoder;
from sklearn.model_selection import train_test_split;

LE = LabelEncoder();

train_seq = pd.read_csv('../DATA/train_seq_all_tanggal_w15m.csv');
train1 = pd.read_csv('../DATA/train1_otp.csv');

train_seq = train_seq.drop(train_seq.columns[0],1);
train_seq['menit_15'] = -1;

dat2 = train1[['name','ADM2_EN','fclass','Agriculture and fisheries','Commercial','Community services','Industrial','Office','Residential','Transport','Area','sum_length','totalPPP','Pop_Total']].drop_duplicates();

train_seq_all2 = train_seq.copy();

train_seq_all2 = pd.merge(train_seq,
                 dat2,
                 on=['name', 'ADM2_EN', 'fclass'],
                 how='left');

train_seq_all2['isWeekend'] = 0;
train_seq_all2.loc[((train_seq_all2['tanggal'] == 'april_6') | (train_seq_all2['tanggal'] == 'april_7')),'isWeekend'] = 1

dat = train_seq_all2.copy();

dat['name'] = LE.fit_transform(dat['name'])
dat['ADM2_EN'] = LE.fit_transform(dat['ADM2_EN'])
dat['fclass'] = LE.fit_transform(dat['fclass'])
dat['hour'] = LE.fit_transform(dat['hour'])
dat['isWeekend'] = LE.fit_transform(dat['isWeekend'])

dat.loc[(dat['X1'] <= 0.1),'X1'] = 0.1;
dat.loc[(dat['X2'] <= 0.1),'X2'] = 0.1;
dat.loc[(dat['X3'] <= 0.1),'X3'] = 0.1;
dat.loc[(dat['X4'] <= 0.1),'X4'] = 0.1;
dat.loc[(dat['X5'] <= 0.1),'X5'] = 0.1;

print(dat['Transport'].isna().sum());

dat.loc[dat['Agriculture and fisheries'].isna(),'Agriculture and fisheries'] = 0;
dat.loc[dat['Transport'].isna(),'Transport'] = 0;

dat = dat.drop(['menit_15','tanggal','name'],1);

subsample = list(map(float, [0.25, 0.5, 0.75]))
max_depth = list(map(int, [3,5,10]))
lambda_ = list(map(float, [0.1,1,10]))
eta = list(map(float, [0.01,0.1,1]))
tuning_grid = pd.DataFrame(list(itertools.product(subsample,max_depth,lambda_,eta)),columns=('subsample','max_depth','reg_lambda','learning_rate'));

target = dat['X5'].copy();
data = dat.drop('X5',1);
dtrain = xgb.DMatrix(data,target);

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=12345);

#################################

import sklearn;

model = xgb.XGBRegressor();
model.fit(X_train,y_train);

preds = model.predict(X_test);
print(sklearn.metrics.r2_score(y_test,preds));
0.5031847752655616

#################################

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import multiprocessing
import sklearn
import json

n_jobs = multiprocessing.cpu_count()-1;

parameters = {'nthread':[4],
              'objective':['binary:logistic'],
              'learning_rate': [0.05], #so called `eta` value
              'max_depth': [6],
              'min_child_weight': [11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [5], #number of trees, change it to 1000 for better results
              'missing':[-999],
              'seed': [1337]}

params_sample = {
	'tree_method': ['exact'],
	'subsample': [0.25, 0.5],
	'learning_rate': [0.01,0.1]
};

params_complete = {
	'tree_method': ['exact'],
	'subsample': [0.25, 0.5, 0.75],
	'max_depth': [3,5,10],
	'reg_lambda': [0.1,1,10],
	'learning_rate': [0.01,0.1,1]
};

xgb_model = xgb.XGBRegressor(
	objective = 'reg:squarederror',
	n_estimators = 100,
	early_stopping_rounds=10
);

clf = GridSearchCV(xgb_model, params_complete, n_jobs=n_jobs,
                   cv=KFold(10,True,12345),
                   scoring='r2',
                   verbose=2, refit=True);
clf.fit(X_train, y_train);

[Parallel(n_jobs=7)]: Done 810 out of 810 | elapsed: 196.2min finished
clf.best_params_
Out[6]:
{'learning_rate': 0.1,
 'max_depth': 10,
 'reg_lambda': 10,
 'subsample': 0.75,
 'tree_method': 'exact'}

pred = clf.predict(X_test);
print(sklearn.metrics.r2_score(y_test,pred));

Out[22]: 0.5133958825622922

#############################################

import numpy as np
import matplotlib.pyplot as plt

r_squared = sklearn.metrics.r2_score(y_test,pred);
plt.scatter(y_test,pred);
plt.xlabel('Actual values');
plt.ylabel('Predicted values');

plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, pred, 1))(np.unique(y_test)));

plt.text(0.6, 0.5, 'R-squared = %0.2f' % r_squared);
plt.show();

#############################################

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import multiprocessing
from scipy import stats
import sklearn

param_dist = {'n_estimators': stats.randint(150, 1000),
              'learning_rate': stats.uniform(0.01, 0.6),
              'subsample': stats.uniform(0.3, 0.9),
              'max_depth': [3, 4, 5, 6, 7, 8, 9],
              'colsample_bytree': stats.uniform(0.5, 0.9),
              'min_child_weight': [1, 2, 3, 4]
             }

params_dist = {
	'tree_method': ['exact'],
	'subsample': stats.uniform(0.3, 0.9),
	'learning_rate': stats.uniform(0.01, 0.6)
};

xgb_model = xgb.XGBRegressor(
	objective = 'reg:squarederror',
	n_estimators = 100,
	early_stopping_rounds=10
);

n_jobs = multiprocessing.cpu_count()-1;

clf = RandomizedSearchCV(xgb_model,
						 params_dist,
						n_iter=10,
						n_jobs=n_jobs,
						 cv=KFold(10,True,12345),
						 random_state=12345,
						scoring='r2',
                   		verbose=2, refit=True);

clf.fit(X_train, y_train);
pred = clf.predict(X_test);
sklearn.metrics.mean_squared_error(y_test,pred);
sklearn.metrics.r2_score(y_test,pred)
0.506275680859126

####################################################
### NOT WORKING YET
####################################################

from hyperopt import hp
from hyperopt.pyll import scope
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# XGB parameters
xgb_reg_params = {
    'learning_rate':    hp.choice('learning_rate',    np.arange(0.05, 0.31, 0.05)),
    'max_depth':        hp.choice('max_depth',        np.arange(5, 16, 1, dtype=int)),
    'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
    'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
    'subsample':        hp.uniform('subsample', 0.8, 1),
    'n_estimators':     100,
    'objective' : 'reg:squarederror'
}
xgb_reg_params2 = {
    'learning_rate':    hp.loguniform('learning_rate',np.log(0.0001), np.log(0.5)) - 0.0001,
    'max_depth':        scope.int(hp.uniform('max_depth',1, 11)),
    'min_child_weight': scope.int(hp.loguniform('min_child_weight',np.log(1), np.log(100))),
    'colsample_bytree': hp.uniform('colsample_bytree',0.5, 1),
    'subsample':        hp.uniform('subsample', 0.8, 1),
    'n_estimators':     100,
    'objective' : 'reg:squarederror'
}

xgb_fit_params = {
    'eval_metric': 'rmse',
    'early_stopping_rounds': 10,
    'verbose': False
}
xgb_para = dict()
xgb_para['reg_params'] = xgb_reg_params2
xgb_para['fit_params'] = xgb_fit_params
xgb_para['loss_func' ] = lambda y, pred: np.sqrt(mean_squared_error(y, pred))

import xgboost as xgb
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials

class HPOpt(object):

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test  = x_test
        self.y_train = y_train
        self.y_test  = y_test

    def process(self, fn_name, space, trials, algo, max_evals):
        fn = getattr(self, fn_name)
        try:
            result = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return result, trials

    def xgb_reg(self, para):
        reg = xgb.XGBRegressor(**para['reg_params'])
        return self.train_reg(reg, para)

    def train_reg(self, reg, para):
        reg.fit(self.x_train, self.y_train,
                eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)],
                **para['fit_params'])
        pred = reg.predict(self.x_test)
        loss = para['loss_func'](self.y_test, pred)
        return {'loss': loss, 'status': STATUS_OK}

obj = HPOpt(X_train, X_test, y_train, y_test)

xgb_opt = obj.process(fn_name='xgb_reg', space=xgb_para, trials=Trials(), algo=tpe.suggest, max_evals=10)

model = xgb.XGBRegressor(
    colsample_bytree = 0.9760113887185847,
    learning_rate = 0.28820148093631176,
    max_depth = 7,
    min_child_weight = 7.619803211294129,
    subsample = 0.9633969560904857,
    n_estimators = 100,
    objective = 'reg:squarederror',
    eval_metric = 'rmse',
    early_stopping_rounds = 10,
    verbose = False
);

clf = model.fit(X_train,y_train);

preds = clf.predict(X_test);

print(r2_score(y_test,preds));
0.5084898349358241

