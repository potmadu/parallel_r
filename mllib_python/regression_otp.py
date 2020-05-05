import pandas as pd;
import numpy as np;
import xgboost as xgb;
import itertools;
from sklearn.preprocessing import LabelEncoder;

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

for iter in range(20:30):
	i = iter-1;
	param = tuning_grid.iloc[[i]].reset_index();
	params = {
			'objective': 'reg:squarederror',
			'eval_metric': 'rmse',
			'tree_method': 'exact'
		}
	params2 = {
		'objective': 'reg:squarederror',
		'eval_metric': 'rmse',
		'tree_method': 'exact',
		'subsample' : 0.75,
		'max_depth' : 10,
		'reg_lambda' : 10,
		'learning_rate' : 1
	}
	params.update(param.transpose().to_dict().get(0));
	params_df = pd.DataFrame(list(params.items()),columns = ['Params','Value']);
	cv = xgb.cv(
		params2,
		dtrain,
		metrics='rmse',
		num_boost_round=100,
		nfold=10,
		seed=12345,
		early_stopping_rounds=10,
		shuffle=True,
		callbacks=[xgb.callback.print_evaluation(show_stdv=True)]
	)

####################

params = {
	'objective': 'reg:squarederror',
	'eval_metric': 'rmse',
	'tree_method': 'exact',
	'subsample': 0.75,
	'max_depth': 10,
	'reg_lambda': 10,
	'learning_rate': 1
}

gridsearch_params = [
    (subsample, max_depth, reg_lambda, learning_rate)
    for subsample in map(float, [0.25, 0.5, 0.75])
    for max_depth in map(int, [3,5,10])
	for reg_lambda in map(float, [0.1,1,10])
	for learning_rate in map(float, [0.01,0.1,1])
]

min_mae = float("Inf")
best_params = None
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_child_weight))    # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=10
    )
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (max_depth,min_child_weight)print("Best params: {}, {}, MAE: {}".format(best_params[0], best