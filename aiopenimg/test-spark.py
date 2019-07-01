load_options = { "table": "data", "keyspace": "production","spark.cassandra.auth.username":"eric","spark.cassandra.auth.password":"cartman123"}

df=spark.read.format("org.apache.spark.sql.cassandra").options(**load_options).load()

