library(AppliedPredictiveModeling);
library(dplyr);

data(abalone);
packages = c('dplyr','AppliedPredictiveModeling');

input_data = abalone;

calcParallel = function(input_data, packages){

  library(doParallel);
  library(foreach);

  cores = detectCores(logical = FALSE)-1;
  cl = makeCluster(cores);
  registerDoParallel(cl, cores=cores);

  chunk.size = ceiling(nrow(input_data)/cores);
  total.chunk.size = cores * chunk.size;
  diff.chunk = total.chunk.size - nrow(input_data);

  added_data = 0;

  if(diff.chunk>0){
    for(i in 1:diff.chunk){
        input_data = rbind(input_data,input_data[1,]);
        added_data=added_data+1;
      }
  }

   res.p = foreach(i=1:cores, .combine='rbind', .packages=packages) %dopar%
   {
      res = matrix(0, nrow=chunk.size, ncol=1)
      for(x in ((i-1)*chunk.size+1):(i*chunk.size)) {
          #res[x - (i-1)*chunk.size,] = distGeo(c(alerts$line_x[x],alerts$line_y[x]),loc);
          res[x - (i-1)*chunk.size,] = colnames(input_data[which.max(input_data[x,])])
      }
      res;
   }

    if(diff.chunk>0){
            res.p = res.p[-((nrow(res.p)-diff.chunk)+1:nrow(res.p)),];
    }

    stopImplicitCluster();
    stopCluster(cl);

    return(res.p);

}

system.time(calcParallel(abalone,packages))

for (i in 1:nrow(abalone)){
  abalone$res = colnames(abalone[which.max(abalone[i,])])
}

