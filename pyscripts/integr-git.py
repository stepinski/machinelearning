import datetime as datetime
import requests as requests
import json
#we go back 15 minutes (1 hour is for timezone?)
nw=datetime.datetime.now() - datetime.timedelta(minutes = 75)
nwstr=nw.strftime("%Y-%m-%dT%H:%M:%SZ")
headers = {'Authorization': 'token e87c979c8bb5414a499593411f7f1939ee355e68'}
url="https://api.github.com/search/issues?q=org:CARLDATA+state:open+created:"+nwstr+"..*"
r = requests.get(url, headers=headers)
output=r.json()
res=output.get("items")
tmp=[]
for r in res:
    tmp.append({'url':r.get('url'),'title':r.get('title'),'body':r.get('body')})

return tmp