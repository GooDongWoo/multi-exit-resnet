import random
import plotly.express as px
import pandas as pd

import matplotlib.pyplot as plt
import pandas as pd
from plotly.subplots import make_subplots

# visualization of result in input resource

#computation_exit=[24, 54, 87, 157]
#expectation accuracy: mAP_list=[45, ]  # 4-exit exist and each has expectation mAP
file_name=file_name_resource_data
resorce_simul=loadResource(file_name)
latency_result=[[0 for _ in range(2)] for _ in range(total_exit)]

for exit in range(total_exit):
    for current_resource in resorce_simul:
        latency_result[exit][0]+=makeLatency(exit,current_resource)
        latency_result[exit][1]+=random.randint(int(mAP_list[exit]+.5)-(mAP_list[exit]//10),int(mAP_list[exit]+.5)+(mAP_list[exit]//10))
    latency_result[exit][1]/=len(resorce_simul)
#my algorithm value latency_result[2][~]
latency_result.append(result_latency_mAP)
#my algorithm value
latency_result

df_result=pd.DataFrame(latency_result,columns=['Total Latency','mAP'],index=['Only Exit0','Only Exit1','Only Exit2','Only Exit3','ALGORITHM\nP_th:75'])
df_result.plot( kind= 'bar' , secondary_y= 'mAP' , rot= 0 ,title='Result of the algorithm simulation(Exit-4)', mark_right=False, grid=True, figsize=(10,7))
plt.show()


df_result=pd.DataFrame(result_data_status,columns=['current_resource','target_exit','DPP','Latency','mAP'])
figures = [
            px.line(df_resource, x=df_resource.index, y="resource",height=1000),
            px.line(df_result, x=df_result.index, y="target_exit",height=1000),#limit y-axis
            px.line(df_result, x=df_result.index, y='mAP',height=1000),
            px.line(df_result, x=df_result.index, y="Latency",height=1000)
    ]

fig = make_subplots(rows=len(figures), cols=1,row_heights=[1000]*len(figures), subplot_titles=['current_resource','target_exit','mAP','Latency'])

for i, figure in enumerate(figures):
    for trace in range(len(figure["data"])):
        fig.append_trace(figure["data"][trace], row=i+1, col=1)
        
fig.show()