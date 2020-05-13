import glob

import pandas as pd;
import numpy as np;
import pickle;
import os;

os.chdir('F:/[Github]/');

#############################
## READ CSV USUAL
#############################

trajectory = pd.read_csv('F:/[Github]/DATA/trajectory_20190405.csv');

#############################
## READ CSV CHUNKING
#############################

import pandas as pd
for chunk in pd.read_csv(<filepath>, chunksize=<your_chunksize_here>)
    do_processing()
    train_algorithm()

#############################################
## READ AND WRITE HD5 : MEMORY LEAK PROBLEMS
#############################################

trajectory_h5 = pd.HDFStore('F:/[Github]/DATA/trajectory_20190405.h5');
trajectory_h5['20190405'] = trajectory;  # save it,

## ERROR : MEMORY LEAK
trajectory = trajectory_h5['20190405']; # retrieve
trajectory_h5.close()

from pandas import DataFrame
from numpy.random import randn
bar = DataFrame(randn(10, 4))
store = HDFStore('test.h5')
store['foo'] = bar   # write to HDF5
bar = store['foo']   # retrieve
store.close()

#############################################
## SPARK
#############################################

from pyspark import sql, SparkConf, SparkContext
from pyspark.sql.functions import countDistinct
import pyspark.sql.functions as f

conf = SparkConf().setAppName("Read_CSV")
sc = SparkContext(conf=conf)
sqlContext = sql.SQLContext(sc)

file = "F:/Github_DATA/trajectory_20190405.csv";
df = sqlContext.read.csv(file,header=True);
print(df.show(10));
#+---+--------------------+----------------+----------------+--------+--------+--------+------+----------+
#|_c0|           driver_id|             lat|            long|accuracy|altitude| bearing| speed| timestamp|
#+---+--------------------+----------------+----------------+--------+--------+--------+------+----------+
#|  1|847bec287cb1f5d01...|      13.8227567|     100.5586571|       8|       0|     173|2.5758|1554429630|
#|  2|37d23926015b1d59b...|      13.7848723|     100.6303533|     3.9|       0|     157|  5.37|1554429630|
#|  3|1b302764dc94a1870...|      13.7571228|     100.5211772|     3.9|       0|     110|  1.33|1554429630|
#|  4|a1537645b0c8d57ae...|      13.7069783|       100.37404|       3|       0|     336|6.0886|1554429630|
#|  5|4cdac58969799737f...|13.7458231550238|100.602288284409|      65|  1.4961|139.2761|     0|1554429630|
#|  6|71b0b1b9a3cb1b00c...|      13.7441713|     100.5411342|   3.748|       0|     137|  0.62|1554429630|
#|  7|d19d33b601bf7d9cb...|      13.7368293|     100.6295206|   3.068|       0|     275|  0.72|1554429630|
#|  8|4feb0bc6536422e3a...|      13.6906258|     100.3438363|  26.651|       0|      66|  1.85|1554429630|
#|  9|81ac2dfdabc13b0a8...|      13.8597782|     100.4115412|       3|       0|     174|5.7114|1554429630|
#| 10|0b355c2953c3928e1...|      13.7216706|     100.5314962|  13.583|       0|     183|  0.43|1554429630|
#+---+--------------------+----------------+----------------+--------+--------+--------+------+----------+
#only showing top 10 rows

print(df.select("driver_id", "lat", "long", "accuracy").show(10));
#+--------------------+----------------+----------------+--------+
#|           driver_id|             lat|            long|accuracy|
#+--------------------+----------------+----------------+--------+
#|847bec287cb1f5d01...|      13.8227567|     100.5586571|       8|
#|37d23926015b1d59b...|      13.7848723|     100.6303533|     3.9|
#|1b302764dc94a1870...|      13.7571228|     100.5211772|     3.9|
#|a1537645b0c8d57ae...|      13.7069783|       100.37404|       3|
#|4cdac58969799737f...|13.7458231550238|100.602288284409|      65|
#|71b0b1b9a3cb1b00c...|      13.7441713|     100.5411342|   3.748|
#|d19d33b601bf7d9cb...|      13.7368293|     100.6295206|   3.068|
#|4feb0bc6536422e3a...|      13.6906258|     100.3438363|  26.651|
#|81ac2dfdabc13b0a8...|      13.8597782|     100.4115412|       3|
#|0b355c2953c3928e1...|      13.7216706|     100.5314962|  13.583|
#+--------------------+----------------+----------------+--------+
#only showing top 10 rows

print(df.groupBy("driver_id").count().show(100));
#+--------------------+-----+
#|           driver_id|count|
#+--------------------+-----+
#|dbfe3293633778e6f...| 2252|
#|0b49b1942970d9c2e...| 2702|
#|00b4be766221fc6d0...| 3859|
#|8a7f12fd56c5b1792...| 2830|
#|2e6bbf32d1a538d90...| 3862|
#|bd905ac7e83e4140a...| 2943|
#|c5f0b5e0d0542c7db...|  817|
#|49eee96b632a38374...| 4771|
#|21b4b8cd9e68b6ccc...| 1450|
#|6c1677f0edebea5d7...|  466|

print(df.select('driver_id').agg(countDistinct('driver_id')).show());
#|count(DISTINCT driver_id)|
#+-------------------------+
#|                    49375|
#+-------------------------+

# add an index column
df = df.withColumn('index', f.monotonically_increasing_id())
df1 = df.sort('index').limit(100)
# sort descending and take 400 rows for df2
df2 = df.sort('index', ascending=False).limit(400)

