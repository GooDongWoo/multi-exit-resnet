import time
import random
import os
import threading
import plotly.express as px
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from plotly.offline import plot
from plotly.subplots import make_subplots
####################################################################################################################################
def increaseY():
    global Y_t,fps_
    Y_t+=fps_             #virtual queue fps:30
    threading.Timer(1, increaseY).start()

def makeResource():
    #making resource
    global min_resource,max_resource,resorce_step,image_num,file_name_resource_data
    resorce_list=[i for i in range(min_resource,max_resource+1,resorce_step)] #length: 111
    #making resource_simulation
    resource_simulation=[0]*image_num
    for each_image in range(image_num):
        if(each_image<(image_num//3)):# early state -> high resource(Idle)
            resource_simulation[each_image]=resorce_list[random.sample(range(-(len(resorce_list)//10),0),k=1)[0]]
        elif((image_num//3)<=each_image<((image_num*2)//3)):# middle state -> low resource
            resource_simulation[each_image]=resorce_list[random.sample(range((len(resorce_list)//10)),k=1)[0]]
        else:# last state -> middle resource
            resource_simulation[each_image]=resorce_list[random.sample(range((len(resorce_list)//3),(len(resorce_list)//3)*2),k=1)[0]]
    with open(file_name_resource_data, 'w+') as file:
        file.write('\n'.join(list(map(str,resource_simulation))))  # '\n' 대신 ', '를 사용하면 줄바꿈이 아닌 ', '를 기준으로 문자열 구분함

def makeDPP(exit,Y_t,p_th):
    global mAP_list
    DPP=(2*Y_t*(p_th-mAP_list[exit]))+((p_th-mAP_list[exit])**2)
    return DPP

def makeLatency(exit,current_resource):
    global computation_exit
    latency=computation_exit[exit]/current_resource
    return latency

def loadResource(file_name):
    resorce_simul=[]
    with open(file_name, "r") as file:
        for i in file:
            resorce_simul.append(int(i.strip()))
    return resorce_simul

def makedataVsearch():
    global total_exit,min_resource,max_resource,resorce_step,file_name_for_Vsearch,V
    df = pd.DataFrame(columns=['exit','DPP','current_resource','latency','p_th','objective'])
    Y_t=0
    for p_th in tqdm(range(min(mAP_list),max(mAP_list)+1),desc=f"p_th",leave=False):
        for exit in range(total_exit):#4번
            DPP=makeDPP(exit,Y_t,p_th)
            for current_resource in range(min_resource*100,max_resource+1,resorce_step*100):#40번 #100Mhz~4Ghz step:100Mhz
                latency=makeLatency(exit,current_resource)
                objective = DPP + (V*latency)
                df.loc[len(df)] = {'p_th' : p_th,'current_resource' : current_resource, 'exit' : exit, 'latency' : latency,'DPP' : DPP,'objective' : objective}
    df.to_csv(file_name_for_Vsearch, index=False)
####################################################################################################################################
#main function
def main():
    global Y_t,result_latency_mAP,result_data_status,total_exit,file_name_resource_data
    resorce_simul=loadResource(file_name_resource_data)
    #increaseY()             # virtual queue fps:30 !!!!!!!!!!!!!!!!!!START!!!!!!!!!!!!!!
    
    for image in range(image_num): #image만큼 iteration process
        current_resource=resorce_simul[image] #dynamic resource state
        #calculate objective function and get target exit number
        objective=infinity
        target_exit=total_exit-1       #initializing target_exit as final exit
        for exit in range(total_exit):
            latency=makeLatency(exit,current_resource)
            DPP=makeDPP(exit,Y_t,p_th)
            if(DPP + (V*latency)<objective):
                objective = DPP + (V*latency)
                target_exit=exit

        # inference step
        #inferenceFunction(latency) # this make latency function, so this can be commentize
        
        # after inference
        inference_result_mAP=random.randint(int(mAP_list[target_exit]+.5)-10,int(mAP_list[target_exit]+.5)+10)
        result_latency_mAP[0]+= makeLatency(target_exit,current_resource)  #total latency
        result_latency_mAP[1]+= inference_result_mAP/image_num #mAP
        result_data_status.append([current_resource,target_exit,makeDPP(target_exit,Y_t,p_th),makeLatency(target_exit,current_resource),inference_result_mAP])
        Y_t=max(Y_t+p_th-inference_result_mAP,0)
        print(f'{image}번째 이미지의 exit: {target_exit}, resource값: {current_resource}, objective값: {objective}')
####################################################################################################################################
#expectation accuracy
mAP_list=[40,60,80,90]  # 4-exit exist and each has expectation mAP
####################################################################################################################################
#constant variables 
infinity=int(1e9)
total_exit=len(mAP_list)            #exit number is 2
####################################################################################################################################
#hyperparameters
V=4*int(1e8)                    # Latency weight constants
p_th=75                 ####### project's key hyperparameter, this parameter determine the custom tendency of exit ##############################################################
image_num=5000          # image number
#fps_=30                 # fps:30
####################################################################################################################################
#layer class
layer_calculation={     # each layer's calculation value
    'init_layer':1,
    'layer1':7,
    'layer2':10,
    'layer3':50,
    'layer4':16,
    'RPN':4,
    'Detector':16,
    'EE':3}
computation_exit=[0]*total_exit 
computation_exit[0]=(layer_calculation['init_layer']+layer_calculation['EE']+layer_calculation['RPN']+layer_calculation['Detector'])
computation_exit[1]=(computation_exit[0]+layer_calculation['layer1']+layer_calculation['EE']+layer_calculation['RPN']+layer_calculation['Detector'])
computation_exit[2]=(computation_exit[1]+layer_calculation['layer2']+layer_calculation['EE']+layer_calculation['RPN']+layer_calculation['Detector'])
computation_exit[3]=(computation_exit[2]+layer_calculation['layer3']+layer_calculation['RPN']+layer_calculation['Detector'])
#computation_exit=[24, 54, 87, 157]
####################################################################################################################################
#resource stat
min_resource=1000000    # min:1Mhz
max_resource=4000000000 # max:4Ghz
resorce_step=1000000    # step:1Mhz
####################################################################################################################################
#variables initialize
target_exit=total_exit-1# initializing target exit as last exit
Y_t=0                   # initialize virtual queue
objective=0             # objective function value
####################################################################################################################################
result_latency_mAP=[0]*2# result_latency_mAP[0]: latency, result_latency_mAP[1]: mAP
result_data_status=[]   # result_data_status[0]: resource, result_data_status[1]: final-exit, result_data_status[2]: dpp, result_data_status[3]: latency
####################################################################################################################################
#file_name
file_name_resource_data='./data/resource_simul_exit4.txt'
file_name_for_Vsearch='./data/simul_exit4.csv'
####################################################################################################################################
#run
if __name__ == "__main__":
    if not (os.path.exists(file_name_resource_data)):
        makeResource()
    main()
    
