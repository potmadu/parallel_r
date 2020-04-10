library(doParallel);
library(foreach);

test;

input_data = read.csv('data/Airports.csv');
packages = c();

calcParallel = function(input_data, packages){

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

  start_time = Sys.time();

   res.p = foreach(i=1:cores, .combine='rbind', .packages=packages) %dopar%
   {
      res = matrix(0, nrow=chunk.size, ncol=1)
      for(x in ((i-1)*chunk.size+1):(i*chunk.size)) {
          res[x - (i-1)*chunk.size,] = distGeo(c(alerts$line_x[x],alerts$line_y[x]),loc);
      }
      res;
   }

    if(diff.chunk>0){
            res.p = res.p[-((nrow(res.p)-diff.chunk)+1:nrow(res.p)),];
    }

    end_time = Sys.time();
    print('elapsed time:');
    print(end_time - start_time);

    stopImplicitCluster();
    stopCluster(cl);

}
