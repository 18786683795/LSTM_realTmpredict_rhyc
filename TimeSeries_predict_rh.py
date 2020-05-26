# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 09:54:18 2019

@author: Lenovo
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
import pymysql
from sqlalchemy import create_engine
from data_processing import data_processing
from datetime import datetime, timedelta


#定义超参数
    
input_size=13
output_size=13
mean=np.array([358.637,13.5976,80.174,16.5052,25.3909,60.5588,9.5699,11.0471,36.8855,6.2935,27.0285,50.9569,20.6291])
std=np.array([362.02239963,26.59734713,92.56434046,25.6864239,51.21534631,71.97984122,21.45158069,18.20996105,51.09030231,17.70573799,36.89085642,74.19714713,34.06744095])

#def load_data(): 
#    print("开始准备特征数据：》》》》》》》》》》》")
#    df=data_processing() 
#    print("数据准备完毕：！！！")   
#    time.sleep(5)
#    print("开始交通流量预测：》》》》》》》》》》》》》》》》》")
#    return df


print("开始准备特征数据：》》》》》》》》》》》")
df=data_processing() 
print("数据准备完毕：！！！")   
time.sleep(2)
print("开始交通流量预测：》》》》》》》》》》》》》》》》》")

def Lstm_forecast(rnn_unit,lr,md):
    tf.reset_default_graph()
    def forecast_Model(lk):
#        df=load_data()
        col = df.columns
        df1=df[df['lk']==lk]
        data=df1[col[3:16]].astype('int').values 
        print(lk+"数据准备完毕：！！！")
        tf.reset_default_graph()
        def get_forecast_data(time_step=6,test_begin=0):
            data_test=data[test_begin:]
        #    mean=np.mean(data_test,axis=0)
        #    std=np.std(data_test,axis=0)
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
        
        
        def prediction(time_step=6):
            print("开始交通流量预测：》》》》》》》》》》》")
            X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
            #Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
            _,_,test_x,test_y=get_forecast_data(time_step)
            pred,_=lstm(X)     
            saver=tf.train.Saver(tf.global_variables())
            with tf.Session() as sess:
                #参数恢复
                base_path = os.getcwd()
                module_file = tf.train.latest_checkpoint(base_path+md) 
                saver.restore(sess, module_file) 
                
                #取训练集最后一行为测试样本。shape=[1,time_step,input_size]
                prev_seq=test_x[-1]
    #            print("prev_seq1:",prev_seq)
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
                dat = data.mean(axis=0)
                
                f = pd.DataFrame((abs(rel_predict).values-dat)/dat).replace(np.inf,0)
                f = f.replace(np.nan,0).convert_objects(convert_numeric=True)
                rel_predict2 = pd.DataFrame(np.where(f>0.05,dat,rel_predict))
                g = pd.DataFrame((abs(rel_predict2).values-dat)/dat).replace(np.inf,0)
                g = g.replace(np.nan,0).convert_objects(convert_numeric=True)
                rel_predict1 = pd.DataFrame(np.where(g<-0.05,dat,rel_predict2))
                
#                fac = (abs(rel_predict).values-dat)/dat
#                func = [lambda fac: 0.15 < fac < 3.5 and -3.5 < fac < -0.15]
#                rel_predict1 = round(pd.DataFrame(np.where(func,dat,rel_predict)))
                
#                f=rel_predict1.values
#                func1=[lambda f: f==0]
#                rel_predict2 = round(pd.DataFrame(np.where(func1,rel_predict1,rel_predict)))
                return rel_predict1
        rel_predict = prediction()
        return rel_predict
    

    def main():
        today = datetime.now()-timedelta(days=1)
        #Tm = (today + timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M")
        Tm = today.strftime("%Y-%m-%d %H:%M")
        sd = Tm[14:16]
        if int(sd) in range(10):
            t=10
        if int(sd) in range(10,20):
            t=20
        if int(sd) in range(20,30):
            t=30
        if int(sd) in range(30,40):
            t=40
        if int(sd) in range(40,50):
            t=50
        if int(sd) in range(50,60):
            t=60    
    
        prediction = []    
        for lk in set(df['lk']):
            yc = forecast_Model(lk)
            col=df.columns
            yc.columns = col[3:16]
            yc['lk'] = lk
            yc['sj'] = Tm[:14]+str(t)
            yc['minut'] = range(1)  ##############
            cols = list(yc)
            cols.insert(0,cols.pop(cols.index('lk')))
            cols.insert(1,cols.pop(cols.index('sj')))
            cols.insert(2,cols.pop(cols.index('minut')))
            yc = yc.loc[:,cols]
            prediction.append(yc)
            
        prediction = pd.concat(prediction)
#        print(prediction)
        
        predt = []
        for lk in set(prediction['lk']):
            dt1 = prediction[prediction['lk']==lk]
            sj2 = [x[:14] for x in dt1['sj']]
    #        sj3 = np.array([str(t+x) for x in range(6)])
#            sj1 = sj2+(dt1['minut']+1).astype('str')+str(0)
#            dt1['sj'] = sj1    ###6个时刻
            dt1['sj'] = yc['sj']  ####1个时刻
            predt.append(dt1)
        predt = pd.concat(predt)
#        print(predt)
        
        print("交通流量预测完毕：》》》》》》》》》》》")
        return predt
    predt0=main()
    return predt0
        #预测结果写入数据库
        #engine=create_engine("mysql+pymysql://root:123456@52.1.123.6:3306/keenIts?charset=utf8")
        #predt.to_sql(name='flow_TimeSeries',con=engine,if_exists='append',index=False,index_label=False)
    
    
def forecast_main():
    predict1 = Lstm_forecast(rnn_unit=10,lr=0.0006,md='/model_10_0.0006/')
    tf.reset_default_graph()
    predict2 = Lstm_forecast(rnn_unit=15,lr=0.0008,md='/model_15_0.0008/')
    tf.reset_default_graph()
    predict3 = Lstm_forecast(rnn_unit=30,lr=0.0005,md='/model_30_0.0005/')
    
    result = round(predict1[predict1.columns[3:]]*0.35+predict2[predict2.columns[3:]]*0.35+predict3[predict3.columns[3:]]*0.3)    
    result.columns = predict1.columns[3:]
    result[['lk','sj','minut']] = predict1[['lk','sj','minut']]
    #交换列顺序
    cols = list(result)
    cols.insert(0,cols.pop(cols.index('lk')))
    cols.insert(1,cols.pop(cols.index('sj')))
    cols.insert(2,cols.pop(cols.index('minut')))
    result = result.loc[:,cols]
#    print("RESULT:",result)
    result.to_csv('result.csv')
    result1=result.drop(['minut'],axis=1)
    result1['point_number']=result['lk']
    result1['time_part']=[x[11:] for x in result1['sj']]
    result1['date']=[x[:10] for x in result1['sj']]
    result2=result1.drop(['sj','lk'],axis=1)
     #交换列顺序
    cols = list(result2)
    cols.insert(0,cols.pop(cols.index('point_number')))
    cols.insert(1,cols.pop(cols.index('date')))
    cols.insert(2,cols.pop(cols.index('time_part')))
    result2 = result2.loc[:,cols]
    result2 = result2.sort_values(by=['point_number'])
    
    print("RESULT:",result2)
    print("预测结果写入数据库》》》》》》》》》》》")
    #预测结果写入数据库
    engine=create_engine("mysql+pymysql://root:123456@52.1.123.6:3306/keenIts?charset=utf8")
    result2.to_sql(name='flow_forecast_TimeSeries',con=engine,if_exists='append',index=False,index_label=False)   
    print("预测完毕！！！！！！！！！！！")
    
    
forecast_main()    





