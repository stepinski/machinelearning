import datetime as datetime
import requests as requests
import json
import schedule

headers = {'Authorization': 'token 25be7a0a0f2f5f4fdee675ac5606801a7696877f'}

url="https://api.github.com/search/issues?q=org:CARLDATA+state:open+created:"+nwstr+"..*"

def getNewIssues(interval=15):
    nw=datetime.datetime.now() - datetime.timedelta(minutes = interval+60)
    nwstr=nw.strftime("%Y-%m-%dT%H:%M:%SZ")
    url="https://api.github.com/search/issues?q=org:CARLDATA+state:open+created:"+nwstr+"..*"
    r = requests.get(url, headers=headers)
    output=r.json()
    res=output.get("items")
    tmp=[]
    for r in res:
        tmp.append({'url':r.get('url'),'title':r.get('title'),'body':r.get('body')})

def transferIssues()

schedule.every(5).minutes.do(getNewIssues, 1, 1000, 100, 1)
