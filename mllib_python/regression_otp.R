# Title     : TODO
# Objective : TODO
# Created by: Work
# Created on: 05/05/2020


##############################################################################
##############################################################################

setwd("E:/Github/GRAB-GIZ/[Analysis]/modelling");

library(xgboost);
library(Matrix);
library(dplyr);
library(forecast);
library(scorer);
library(Metrics);
library(multiROC);
library(caret);
library(DescTools);
library(e1071);

set.seed(12345);

load("train_seq_all_tanggal_w15m.RData");
train1 = read.csv("train1_otp.csv",stringsAsFactors=FALSE);

train_seq$menit_15 = -1;

dat2 = train1 %>%
distinct(name,ADM2_EN,fclass,Agriculture.and.fisheries,Commercial,Community.services,Industrial,Office,Residential,Transport,Area,sum_length,totalPPP,Pop_Total) %>%
as.data.frame();

train_seq_all2 = train_seq %>%
left_join(dat2,by=c("name","ADM2_EN","fclass")) %>%
as.data.frame();

train_seq_all2$isWeekend = 0;
train_seq_all2$isWeekend[train_seq_all2$tanggal=="april_6" | train_seq_all2$tanggal=="april_7"] = 1;

dat = train_seq_all2;

dat$name = as.factor(dat$name);
dat$ADM2_EN = as.factor(dat$ADM2_EN);
dat$fclass = as.factor(dat$fclass);
dat$hour = as.factor(dat$hour);
dat$isWeekend = as.factor(dat$isWeekend);

dat$X1[dat$X1<=0.1] = 0.1;
dat$X2[dat$X2<=0.1] = 0.1;
dat$X3[dat$X3<=0.1] = 0.1;
dat$X4[dat$X4<=0.1] = 0.1;
dat$X5[dat$X5<=0.1] = 0.1;

dat$Agriculture.and.fisheries[is.na(dat$Agriculture.and.fisheries)] = 0;
dat$Transport[is.na(dat$Transport)] = 0;

dat$menit_15 = NULL;
dat$tanggal = NULL;

dat$name = NULL;

load_grid_hyperparameters2 = function(){

	searchGridSubCol = expand.grid(subsample = c(0.25, 0.5, 0.75),
	                                max_depth = c(3, 5, 10),
	                                lambda = c(0.1, 1, 10),
	                                eta = c(0.01, 0.1, 1)
	)

	return(searchGridSubCol);

}

tuning_grid = load_grid_hyperparameters2();

target = dat$X5;

df = sparse.model.matrix(X5~.-1, data = dat);
dtrain = xgb.DMatrix(df,label=target)
dtest = xgb.DMatrix(df)

iter=1;
for(i in 20:30){

	print(i);

	for(j in 1:10){

		set.seed(j);

		cv = xgb.cv(data = dtrain, nrounds = 100, nthread = 7,
			stratified=TRUE,
			nfold = 10,
			tuning_grid[i,],
			early_stopping_rounds=10,
			tree_method="exact",
			objective = "reg:linear",
			prediction=TRUE,
			verbose=FALSE);

		cv$evaluation_log

		preds = cv$pred;
		r2 = r2_score(target, preds);
		kor = cor(target, preds);
		rmse = rmse(target, preds);

		roc_result = data.frame(i,j,r2,kor,rmse);

		if(iter==1){
			output= roc_result;
			iter=2;
		} else{
			output = rbind(output,roc_result);
		}

	}

}

> output %>% group_by(i) %>% summarise(rata_r2=mean(r2),rata_kor=mean(kor),rata_rmse=mean(rmse))
# A tibble: 11 x 4
       i rata_r2 rata_kor rata_rmse
   <int>   <dbl>    <dbl>     <dbl>
 1    20   0.496    0.680     0.217
 2    21   0.496    0.679     0.217
 3    22   0.504    0.692     0.216
 4    23   0.504    0.691     0.215
 5    24   0.504    0.691     0.215
 6    25   0.512    0.699     0.215
 7    26   0.511    0.699     0.215
 8    27   0.511    0.699     0.215
 9    28   0.459    0.701     0.144 V
10    29   0.457    0.701     0.144
11    30   0.456    0.701     0.144

d = density(preds[,1]);
e = density(preds[,2]);

plot(e,col='red');
lines(d,col='black');

title('');

legend("bottomleft",
  legend = c("Actual", "Prediction"),
  col = c('black','red'),
  lwd = c(2,2),
  bty = "n",
  pt.cex = 2,
  cex = 1.2,
  text.col = "black",
  horiz = F ,
  inset = c(0.05, 0.2))

