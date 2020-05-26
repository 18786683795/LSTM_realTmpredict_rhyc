# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:31:10 2019

@author: Lenovo
"""

from datetime import datetime,timedelta
import time
import schedule
from TimeSeries_predict_rh import *
#from real_error import *

#每天8点定时启动
def main():
    schedule.every(10).minutes.do(forecast_main)
    schedule.every(10).minutes.do(forecast_real)
    while True:
        schedule.run_pending()
        time.sleep(10)
        
if __name__ == '__main__':
    main()
    
    
import os
import sys
pwd=os.path.dirname(os.path.realpath('main.py'))
print(pwd)
print(sys.path.append(pwd+"../"))