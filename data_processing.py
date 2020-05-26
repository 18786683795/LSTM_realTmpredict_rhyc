# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:26:53 2019

@author: Lenovo
"""

import pymysql
import sys
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

def data_processing():
    today = datetime.now() #取今天
    yeday = (today - timedelta(days=3)).strftime("%Y-%m-%d") #取昨天，因为只能拿到昨天的数据
    dates = (yeday)
    con = pymysql.connect('52.1.123.6','root','123456','keenIts')
    sql = "select * from t_transportation_flow WHERE point_number in ('#GS002','#GS004','#GS005','#GS006','#GS007',\
    '#GS008','#GS009','#GS010','#GS011','#GS012','#GS017','#GS024','#GS031','#GS033','#GS035') and date in \
    (\"%s\")"%(dates)
    
    data = pd.read_sql(sql,con)
    con.close()
    
    xq = []
    tm = data['date']
    for i in tm:
        dt = pd.to_datetime(i)
        dt1 = dt.strftime('%A')
        xq.append(dt1)
        
    tm = [int(x[0:2]) for x in data['time_part']]
    data['Tm'] = tm
    data['xq'] = xq
    data['rq'] = data.date
    
    size_mapping = {
                   'Monday': str(1),
                   'Tuesday': 2,
                   'Wednesday': 3,
                   'Thursday':4,
                   'Friday':5,
                   'Saturday':6,
                   'Sunday':7}
    data['xq'] = data['xq'].map(size_mapping)
    
    data['flow_e'] = data['flow_e_l'] + data['flow_e_s'] + data['flow_e_r']
    data['flow_w'] = data['flow_w_l'] + data['flow_w_s'] + data['flow_w_r']
    data['flow_s'] = data['flow_s_l'] + data['flow_s_s'] + data['flow_s_r']
    data['flow_n'] = data['flow_n_l'] + data['flow_n_s'] + data['flow_n_r']
    
    #10分钟的交通流量
    sjd = [int(x[12:14]) for x in data['time_part']]
    data['sjd'] = sjd
    
    size_sjd = {
        5:1,
        10:1,
        15:2,
        20:2,
        25:3,
        30:3,
        35:4,
        40:4,
        45:5,
        50:5,
        55:6,
        0:6
    }
    data['sjd'] = data['sjd'].map(size_sjd)
    
#    Tm = datetime.now().strftime("%Y-%m-%d %H:%M")
#    th = Tm[11:13]
#    sd = Tm[14:16]
#    
#    if (int(th[0]==0 and int(th[1]==0))):
#        th = 24
#    else:
#        th = int(th)
#    
#    t = 0
#    if (int(sd[0]) == 0 and int(sd[1]) in range(5)) | (int(sd[0]) == 0 and int(sd[1]) in range(6,10)):
#        t += 1
#    if (int(sd[0]) == 1 and int(sd[1]) in range(5)) | (int(sd[0]) == 1 and int(sd[1]) in range(6,10)):
#        t += 2
#    if (int(sd[0]) == 2 and int(sd[1]) in range(5)) | (int(sd[0]) == 2 and int(sd[1]) in range(6,10)):
#        t += 3
#    if (int(sd[0]) == 3 and int(sd[1]) in range(5)) | (int(sd[0]) == 3 and int(sd[1]) in range(6,10)):
#        t += 4
#    if (int(sd[0]) == 4 and int(sd[1]) in range(5)) | (int(sd[0]) == 4 and int(sd[1]) in range(6,10)):
#        t += 5
#    if (int(sd[0]) == 5 and int(sd[1]) in range(5)) | (int(sd[0]) == 5 and int(sd[1]) in range(6,10)):
#        t += 6
#
#    data1 = data[(data['sjd']==t)&(data['time_hour']==th)|(data['time_hour']==th-1)]
    
    
    #取当前时间所在10分钟的前6个10分钟
    def rq_zh(d):
    #     today = datetime.now() #取今天
        today = datetime.now()-timedelta(days=1)
        Tm = (today - timedelta(minutes=d)).strftime("%Y-%m-%d %H:%M")
        th = Tm[11:13]
        sd = Tm[14:16]
        
        if (int(th[0]==0 and int(th[1]==0))):
            th = 24
        else:
            th = int(th)
        
        t = 0
        if (int(sd[0]) == 0 and int(sd[1]) in range(5)) | (int(sd[0]) == 0 and int(sd[1]) in range(6,10)):
            t += 1
        if (int(sd[0]) == 1 and int(sd[1]) in range(5)) | (int(sd[0]) == 1 and int(sd[1]) in range(6,10)):
            t += 2
        if (int(sd[0]) == 2 and int(sd[1]) in range(5)) | (int(sd[0]) == 2 and int(sd[1]) in range(6,10)):
            t += 3
        if (int(sd[0]) == 3 and int(sd[1]) in range(5)) | (int(sd[0]) == 3 and int(sd[1]) in range(6,10)):
            t += 4
        if (int(sd[0]) == 4 and int(sd[1]) in range(5)) | (int(sd[0]) == 4 and int(sd[1]) in range(6,10)):
            t += 5
        if (int(sd[0]) == 5 and int(sd[1]) in range(5)) | (int(sd[0]) == 5 and int(sd[1]) in range(6,10)):
            t += 6
        return th,t
    dats = []
    for i in [10,20,30,40,50,60]:
        dat = rq_zh(i)
        dats.append(dat)
#    data1 = data[(data['time_hour'].isin([i[0]+1 for i in dats]))&(data['sjd'].isin([i[1] for i in dats]))]
    data1 = [] 
    for i in dats:
        dat1 = data[(data['time_hour'].isin([i[0],i[0]+1]))&(data['sjd']==i[1])]
        data1.append(dat1)
    data1 = pd.concat(data1)
 
    ds = []
    for r in set(data1['date']):
        d1 = data1[data1['date']==r]
        for lk in set(d1['point_number']):
            d2 = d1[d1['point_number']==lk]
            for t in set(d2['Tm']):
                d3 = d2[d2['Tm']==t]
                for sj in set(d3['sjd']):
                    d4 = d3[d3['sjd']==sj]
                    
                    def zj(col):
                        zz = d4[col].sum()
                        return zz
                     
                    flow_all = zj('flow_all')   
                    flow_e_l = zj('flow_e_l')
                    flow_e_s = zj('flow_e_s')
                    flow_e_r = zj('flow_e_r')
                    flow_w_l = zj('flow_w_l')
                    flow_w_s = zj('flow_w_s')
                    flow_w_r = zj('flow_w_r')
                    flow_s_l = zj('flow_s_l')
                    flow_s_s = zj('flow_s_s')
                    flow_s_r = zj('flow_s_r')
                    flow_n_l = zj('flow_n_l')
                    flow_n_s = zj('flow_n_s')
                    flow_n_r = zj('flow_n_r')
                    fl_e = zj('flow_e')
                    fl_w = zj('flow_w')
                    fl_s = zj('flow_s')
                    fl_n = zj('flow_n')
    
                    dsum = round(pd.DataFrame([flow_all,flow_e_l,flow_e_s,flow_e_r,flow_w_l,flow_w_s,
                                               flow_w_r,flow_s_l,flow_s_s,flow_s_r,flow_n_l,
                                               flow_n_s,flow_n_r,fl_e,fl_w,fl_s,fl_n]))
    
                    dsum1 = pd.DataFrame(dsum).T
                    dsum1.columns = ['flow_all','flow_e_l','flow_e_s','flow_e_r','flow_w_l','flow_w_s',
                                               'flow_w_r','flow_s_l','flow_s_s','flow_s_r','flow_n_l',
                                               'flow_n_s','flow_n_r','flow_e','flow_w','flow_s','flow_n']
                    dsum1['Tm'] = t
                    dsum1['lk'] = lk
                    dsum1['rq'] = r
                    dsum1['sjd'] = sj
                    ds.append(dsum1)
    
    try:
        lk = pd.concat(ds,ignore_index=True)
    except ValueError:
        print(" raise ValueError('No objects to concatenate')")
    else:
        print("objects to concatenate success")
        

        lk['time_p'] = lk['Tm'].astype(str) + ":" + lk['sjd'].astype(str) + str(0)

        lk['Time slice'] = lk['rq'].astype(str) + ":" + lk['time_p']
        
        xq = []
        tm = lk['rq']
        for i in tm:
            dt = pd.to_datetime(i)
            dt1 = dt.strftime('%A')
            xq.append(dt1)
            
        lk['xq'] = xq
        
        size_mapping = {
                       'Monday': str(1),
                       'Tuesday': 2,
                       'Wednesday': 3,
                       'Thursday':4,
                       'Friday':5,
                       'Saturday':6,
                       'Sunday':7}
        lk['xq'] = lk['xq'].map(size_mapping)
        
        lk1 = lk
        lk1['sj'] = lk1['rq'].astype('str')+" "+lk1['time_p']
        #交换列顺序
        cols = list(lk)
        cols.insert(0,cols.pop(cols.index('lk')))
        cols.insert(1,cols.pop(cols.index('sj')))
        cols.insert(2,cols.pop(cols.index('sjd')))
        lk1 = lk1.loc[:,cols]
        tz = lk1[lk1.columns[:20]]
        return tz



#取当前时间所在10分钟的前6个10分钟
#def rq_zh(d):
#    from datetime import datetime, timedelta
##     today = datetime.now() #取今天
#    today = datetime.now()
#    Tm = (today - timedelta(minutes=d)).strftime("%Y-%m-%d %H:%M")
#    th = Tm[11:13]
#    sd = Tm[14:16]
#    
#    if (int(th[0]==0 and int(th[1]==0))):
#        th = 24
#    else:
#        th = int(th)
#    
#    t = 0
#    if (int(sd[0]) == 0 and int(sd[1]) in range(5)) | (int(sd[0]) == 0 and int(sd[1]) in range(6,10)):
#        t += 1
#    if (int(sd[0]) == 1 and int(sd[1]) in range(5)) | (int(sd[0]) == 1 and int(sd[1]) in range(6,10)):
#        t += 2
#    if (int(sd[0]) == 2 and int(sd[1]) in range(5)) | (int(sd[0]) == 2 and int(sd[1]) in range(6,10)):
#        t += 3
#    if (int(sd[0]) == 3 and int(sd[1]) in range(5)) | (int(sd[0]) == 3 and int(sd[1]) in range(6,10)):
#        t += 4
#    if (int(sd[0]) == 4 and int(sd[1]) in range(5)) | (int(sd[0]) == 4 and int(sd[1]) in range(6,10)):
#        t += 5
#    if (int(sd[0]) == 5 and int(sd[1]) in range(5)) | (int(sd[0]) == 5 and int(sd[1]) in range(6,10)):
#        t += 6
#    return th,t
#dats = []
#for i in [10,20,30,40,50,60]:
#    dat = rq_zh(i)
#    dats.append(dat)
#
#[i[1] for i in dats]



























