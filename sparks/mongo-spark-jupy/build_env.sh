
echo  "Starting docker ."

docker-compose up -d --build

echo  "Configuring the MongoDB ReplicaSet."

docker-compose exec mongo1 /usr/bin/mongo --eval "if (rs.status()['ok'] == 0) { rsconf = { _id : 'rs0', members: [ { _id : 0, host : 'mongo1:27017', priority: 1.0 }, { _id : 1, host : 'mongo2:27017', priority: 0.5 }, { _id : 2, host : 'mongo3:27017', priority: 0.5 } ] }; rs.initiate(rsconf); } rs.conf();"

echo  "Uploading test data into Stocks database"

docker-compose exec mongo1 apt-get update

docker-compose exec mongo1 apt-get install wget

docker-compose exec mongo1 wget https://github.com/RWaltersMA/mongo-spark-jupyter/raw/master/Source.bson

docker-compose exec mongo1 /usr/bin/mongorestore Source.bson -h rs0/mongo1:27017,mongo2:27018,mongo3:27019 -d Stocks -c Source --drop

echo  "starting Jypterlab"

docker exec -it jupyterlab  /opt/conda/bin/jupyter server list

#echo  NOTE: When running on Windows change 0.0.0.0 to localhost in the URL above

echo "Spark Master - http://localhost:8080 Spark Worker 1 Spark Worker 2 MongoDB Replica Set - port 27017-27019 "

echo "=============================================================================================================="

#Press <Escape> to quit"

#Wait-KeyPress
 
# if we don't specify -v then issue this one -> docker-compose exec mongo1 /usr/bin/mongo localhost:27017/SparkDemo --eval "db.dropDatabase()"

#docker-compose down  -v

# note: we use a -v to remove the volumes, else you'll end up with old data
