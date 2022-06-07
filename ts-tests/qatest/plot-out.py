import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("QAQCvelocity.csv")
print(data.head())
# data.set_index("time", inplace=True)
import plotly.graph_objects as go
import numpy as np

# x = np.arange(10)

# fig = go.Figure(data=go.Scatter(data))
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.time, y=data.value,
                    mode='lines',
                    name='velocity'))

fig.add_trace(go.Scatter(x=data.time, y=data.flag,
                    mode='lines',
                    name='anomalies'))
fig.show()
