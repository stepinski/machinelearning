from pyspark import SparkContext, SparkConf
conf = SparkConf().setAppName('selectas').setMaster('spark://10.150.1.18:7077')
sc = SparkContext(conf=conf)

#(it works on local cassandra .18 with pyspark  ./bin/pyspark --packages com.datastax.spark:spark-cassandra-connector_2.11:2.3.2)
# we need to change configs on all workers to use proper cassandra connector

from pyspark.sql import SQLContext

sqlContext = SQLContext(sc)

load_options = { "table": "data", "keyspace": "production","spark.cassandra.auth.username":"eric","spark.cassandra.auth.password":"cartman123"}

df=sqlContext.read.format("org.apache.spark.sql.cassandra").option("spark.cassandra.connection.host", "10.150.1.80").options(**load_options).load().where( "channel ='fdafdsa'").cache()

df.count()
