Building and deploying the sample
Once you have recreated the sample code files (or used the files in the sample folder) you’re ready to build and deploy the sample app.

Use Docker to build the sample code into a container. To build and push with Docker Hub, run these commands replacing {username} with your Docker Hub username:

# Build the container on your local machine
docker build -t {username}/helloworld-rserver .

# Push the container to docker registry
docker push {username}/helloworld-rserver
After the build has completed and the container is pushed to docker hub, you can deploy the app into your cluster. Ensure that the container image value in service.yaml matches the container you built in the previous step. Apply the configuration using kubectl:

kubectl apply --filename service.yaml
Now that your service is created, Knative performs the following steps:

Create a new immutable revision for this version of the app.
Network programming to create a route, ingress, service, and load balance for your app.
Automatically scale your pods up and down (including to zero active pods).
Run the following command to find the domain URL for your service:

kubectl get ksvc helloworld-r  --output=custom-columns=NAME:.metadata.name,URL:.status.url
Example:

NAME                URL
helloworld-r    http://helloworld-r.default.1.2.3.4.xip.io
Now you can make a request to your app and see the result. Replace the URL below with the URL returned in the previous command.

curl http://helloworld-rserver.default.1.2.3.4.xip.io
Example:

curl http://helloworld-rserver.default.1.2.3.4.xip.io
[1] "Hello R Sample v1!"
Note: Add -v option to get more detail if the curl command failed.


source:https://knative.dev/community/samples/serving/helloworld-rserver/