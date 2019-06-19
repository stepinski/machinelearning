#!/usr/bin/env python3
''' Query data points from KairosDB database using REST API through requests module.
  
    Please, check out the documentation on the KairosDB website:
      http://kairosdb.github.io/docs/build/html/restapi/QueryMetrics.html
    @author Fernando Paladini <fnpaladini@gmail.com>
'''

import requests
import json

kairosdb_server = "http://localhost:8080"

# Simple test
query = {
 "start_absolute": 1104537615,
   "end_relative": {
       "value": "1",
       "unit": "years"
   },
   "metrics": [
       {
           "name": "testdata"
       }
   ]
}
response = requests.post(kairosdb_server + "/api/v1/datapoints/query", data=json.dumps(query))
print("Status code: %d" % response.status_code)
print("JSON response:")
print(response.json())

