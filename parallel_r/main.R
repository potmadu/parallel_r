library(doParallel);
library(foreach);

calc_distance_parallel = function(alerts,loc){

cores = detectCores(logical = FALSE)-1;
cl = makeCluster(cores);
registerDoParallel(cl, cores=cores);

output = alerts;

chunk.size = ceiling(nrow(alerts)/cores);
total.chunk.size = cores * chunk.size;
diff.chunk = total.chunk.size - nrow(alerts);

added_data = 0;

if(diff.chunk>0){
    for(i in 1:diff.chunk){
        alerts = rbind(alerts,alerts[1,]);
        added_data=added_data+1;
    }
}

start_time = Sys.time();

 res2.p = foreach(i=1:cores, .combine='rbind', .packages='geosphere') %dopar%
 {
    res = matrix(0, nrow=chunk.size, ncol=1)
    for(x in ((i-1)*chunk.size+1):(i*chunk.size)) {
        res[x - (i-1)*chunk.size,] = distGeo(c(alerts$line_x[x],alerts$line_y[x]),loc);
    }
    res;
 }

end_time = Sys.time();
end_time - start_time;

if(diff.chunk>0){
        res2.p = res2.p[-((nrow(res2.p)-diff.chunk)+1:nrow(res2.p)),];
}

output$jarak = res2.p;

stopImplicitCluster();
stopCluster(cl);

return(output);

}

alerts_before_dist = calc_distance_parallel(alerts_before,pasar_tanahabang);
alerts_after_dist = calc_distance_parallel(alerts_after,pasar_tanahabang);