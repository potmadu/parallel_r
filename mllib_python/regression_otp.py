import pandas as pd;
import numpy as np;

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

dat['name'] = dat['name'].astype('category')
dat['ADM2_EN'] = dat['ADM2_EN'].astype('category')
dat['fclass'] = dat['fclass'].astype('category')
dat['hour'] = dat['hour'].astype('category')
dat['isWeekend'] = dat['isWeekend'].astype('category')

dat.loc[(dat['X1'] <= 0.1),'X1'] = 0.1;
dat.loc[(dat['X2'] <= 0.1),'X2'] = 0.1;
dat.loc[(dat['X3'] <= 0.1),'X3'] = 0.1;
dat.loc[(dat['X4'] <= 0.1),'X4'] = 0.1;
dat.loc[(dat['X5'] <= 0.1),'X5'] = 0.1;

print(dat['Transport'].isna().sum());

dat.loc[dat['Agriculture and fisheries'].isna(),'Agriculture and fisheries'] = 0;
dat.loc[dat['Transport'].isna(),'Transport'] = 0;

dat = dat.drop(['menit_15','tanggal','name'],1);


