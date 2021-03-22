import datetime as datetime
import requests as requests
import json
#we go back 15 minutes (1 hour is for timezone?)
nw=datetime.datetime.now() - datetime.timedelta(minutes = 75)
nwstr=nw.strftime("%Y-%m-%dT%H:%M:%SZ")
headers = {'Authorization': 'token 25be7a0a0f2f5f4fdee675ac5606801a7696877f'}
url="https://api.github.com/search/issues?q=org:CARLDATA+state:open+created:"+nwstr+"..*"
r = requests.get(url, headers=headers)
output=r.json()
res=output.get("items")
tmp=[]
for r in res:
    tmp.append({'url':r.get('url'),'title':r.get('title'),'body':r.get('body')})

return tmp