import pandas as pd
data = pd.read_csv("velocity.csv")
print(data.head())
data[data['time']>'2021-01-20T14:05:00'].to_csv('test_anom2.csv',index=False)