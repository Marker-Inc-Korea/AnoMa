import requests
from time import sleep
import datetime
import numpy as np
datetime.datetime.now()
import gc
import os
from environs import Env
import logging
import logging.handlers


import pandas as pd
import numpy as np
import math, time
import os
import sys
from scipy import interpolate

from influxdb import InfluxDBClient
# import pdb


# def getxy(hour):
#     x = np.sin((180 - hour * 15)/180 * np.pi)
#     y = np.cos((180 - hour * 15)/180 * np.pi)
#     return x, y

def time_value(t):
    hour = t.hour
    minute = t.minute/60
    second = t.second/(60*60)
    print(hour, minute, second)
    result = hour + minute + second
    # print(result)
    result = result *15
    print(result)
    
    x_ = np.sin((180 - result * 15)/180 * np.pi)
    y_ = np.cos((180 - result * 15)/180 * np.pi)
    
    # x_, y_ = getxy(result)
    return x_, y_

# logging.basicConfig(filename='./log/test.log', level = logging.DEBUG)


def random_gen(url, headers,job_name, instance, instance_name, data_format, data_value):
    data = data_format % (*data_value,)
    print(data)
    r = requests.post('{url}/metrics/job/{job_name}/{instance}/{instance_name}'.format(url = url,
                                                                                   job_name = job_name,
                                                                                   instance = instance,
                                                                                   instance_name = instance_name),
                  data = data,
                  headers=headers)
    print("INFO - {} - {} - {} ".format(datetime.datetime.now(), r, data_value))
    gc.collect()

if __name__ == "__main__":
    env = Env()
    env.read_env()
    ANOMA_IP = env('ANOMA_IP')
    print(ANOMA_IP)
    url = 'http://{ip}:6076'.format(ip=ANOMA_IP)
    job_name = 'some_job'
    instance = 'instance'
    instance_name = 'some_instance'
    headers = {'X-Requested-With': 'Python requests', 'Content-type': 'text/xml'}
    
    data_format = """some_metric{label="val1"} %s\n another_metric %s\n"""
    seq_len = 1000
    while 1:
#         metric_1 = np.random.normal(20, 3, 1000)
#         metric_2 = np.random.normal(200, 3, 1000)
        
        for k_idx in range(seq_len):
            sleep(0.5)
            c_time = datetime.datetime.now()
            x_, y_ = time_value(c_time)
            if k_idx==500:
#                 data_value = [x_*100, y_*100]
                
                data_value = [x_, y_]
#                 data_value = [metric_1[k_idx]*100,metric_2[k_idx]*100 ]
            else:
                data_value = [x_, y_]
#                 data_value = [metric_1[k_idx],metric_2[k_idx] ]
            
            random_gen(url, headers,  job_name, instance, instance_name, data_format, data_value)
        gc.collect()
            
    
    