# -*- coding: utf-8 -*-
"""
Created on Tue May 29 16:17:13 2018
https://blog.csdn.net/mylove0414/article/details/56969181
@author: admin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sklearn.metrics as metrics
import datetime
import time
import schedule
from data_processing import data_processing

#定义超参数
rnn_unit=15       
input_size=13
output_size=13
lr=0.0008         

#f=open('./dataset/tzz_10mint.csv') 
#df=pd.read_csv(f).tail(10000).fillna(0)    
#data=df.iloc[:,3:].values  

print("开始准备特征数据：》》》》》》》》》》》")
df=data_processing()    
cols = df.columns
data=df[cols[3:16]].astype('int').values 
print("数据准备完毕：！！！")


tf.reset_default_graph()
def get_forecast_data(time_step=10,test_begin=0):
    data_test=data[test_begin:]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  #标准化
    size=(len(normalized_test_data)+time_step-1)//time_step
    test_x,test_y=[],[]  
    for i in range(size-1):
       x=normalized_test_data[i:i+time_step,:13]
       y=normalized_test_data[i:i+time_step,:13]
       test_x.append(x.tolist())
       test_y.extend(y)
#    test_x.append((normalized_test_data[(i+1)*time_step:,:5]).tolist())
#    test_y.extend((normalized_test_data[(i+1)*time_step:,:5]).tolist())
    return mean,std,test_x,test_y


#定义权重及偏差
weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,13]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[13,]))
       }


def lstm(X):     
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']  
    input=tf.reshape(X,[-1,input_size])  
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit,reuse=tf.AUTO_REUSE)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  
    output=tf.reshape(output_rnn,[-1,rnn_unit]) 
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states


def prediction(time_step=10):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    #Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    _,_,test_x,test_y=get_forecast_data(time_step)
    mean=np.array([358.637,13.5976,80.174,16.5052,25.3909,60.5588,9.5699,11.0471,36.8855,6.2935,27.0285,50.9569,20.6291])
    std=np.array([362.02239963,26.59734713,92.56434046,25.6864239,51.21534631,71.97984122,21.45158069,18.20996105,51.09030231,17.70573799,36.89085642,74.19714713,34.06744095])
    pred,_=lstm(X)     
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        base_path = os.getcwd()
        module_file = tf.train.latest_checkpoint(base_path+'/model/') 
        saver.restore(sess, module_file) 
        
        #取训练集最后一行为测试样本。shape=[1,time_step,input_size]
        prev_seq=test_x[-1]
#        print("prev_seq1:",prev_seq)
        predict=[]
        #得到之后100个预测结果
        for i in range(1):
            next_seq=sess.run(pred,feed_dict={X:[prev_seq]})
            predict.append(pd.DataFrame(next_seq[-1]).T)
            #每次得到最后一个时间步的预测结果，与之前的数据加在一起，形成新的测试样本
            prev_seq=np.vstack((prev_seq[1:],next_seq[-1]+((np.random.uniform(-5,8,[1,13])-mean)/std)))
#            rel_predict=predict[-1]*std[:5]+mean[:5]
        rel_predict = []
        for i in predict*std[:13]+mean[:13]:
            predict1 = i.reshape(13)
            rel_predict.append(abs(predict1))
        rel_predict=round(pd.DataFrame(rel_predict))
        print(rel_predict)
        


prediction()





#schedule.every(1).minutes.do(prediction)
#while True:
#    schedule.run_pending()
#    time.sleep(5)







#def main(h=10, m=27):
#    '''h表示设定的小时，m为设定的分钟'''
#    while True:
#        # 判断是否达到设定时间，例如23:00
#        while True:
#            now = datetime.datetime.now()
#            # 到达设定时间，结束内循环
#            if now.hour==h and now.minute==m:
#                break
#            # 不到时间就等20秒之后再次检测
#            time.sleep(20)
#        # 做正事，一天做一次
#        prediction()
#if __name__ == '__main__':
#    main()



