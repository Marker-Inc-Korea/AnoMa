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


env = Env()
env.read_env()
ANOMA_IP = env('ANOMA_IP')


# Influx DB basic information is taken from Configuration.
DATABASE_PORT = env('INFLUX_DB_PORT')
DATABASE_NAME = 'prometheus'
DATABSAE_USERNAME = env('INFLUX_DB_USER_NAME')
DATABSAE_PASSWORD = env('INFLUX_DB_PASSWORD')


db_client = InfluxDBClient(ANOMA_IP, DATABASE_PORT, DATABSAE_USERNAME,DATABSAE_PASSWORD,  DATABASE_NAME)


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


def random_gen( table, column, time, data_value):
    
    json_body = {}
    json_body["measurement"] = table
    json_body["label"] = column
    json_body["time"] = time
    json_body["fields"] = {"value" : data_value}
    print("INFO - {} - {} - {} - {} ".format(time, table,column, data_value ))
#     print(json_body)
#     quit()
    return json_body

if __name__ == "__main__":
    s
    table_name = ['cpu_load_short','cpu_load_short2']
    col_name = {"cpu_load_short" : ['val11'], "cpu_load_short2" : ['val22']}
#     job_name = 'some_job'
#     instance = 'instance'
#     instance_name = 'some_instance'

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
            write_points = []
            for t_idx, table_ in enumerate(table_name):
                print(table_)
                for col_ in col_name[table_]:
                    json_body = random_gen( table_, col_, c_time, data_value[t_idx])
                    write_points.append(json_body)
            print(write_points)
#             quit()
            db_client.write_points(write_points, time_precision ='u')
            
    
    