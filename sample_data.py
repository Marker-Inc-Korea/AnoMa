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


# logging.basicConfig(filename='./log/test.log', level = logging.DEBUG)


def random_gen(url, headers,job_name, instance, instance_name, data_format, data_value):
    data = data_format % (*data_value,)
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
    PUSH_PORT = env('PUSH_GATE_WAY_PORT')
    print(ANOMA_IP)
    url = 'http://{ip}:{push_port}'.format(ip=ANOMA_IP, push_port = PUSH_PORT)
    job_name = 'some_job'
    instance = 'instance'
    instance_name = 'some_instance'
    headers = {'X-Requested-With': 'Python requests', 'Content-type': 'text/xml'}
    
    data_format = """some_metric{label="val1"} %s\n another_metric %s\n"""
    
    while 1:
        metric_1 = np.random.normal(20, 3, 1000)
        metric_2 = np.random.normal(200, 3, 1000)
        
        for k_idx in range(len(metric_1)):
            sleep(0.05)
            if k_idx==500:
                data_value = [metric_1[k_idx]*100,metric_2[k_idx]*100 ]
            else:
                data_value = [metric_1[k_idx],metric_2[k_idx] ]
            
            random_gen(url, headers,  job_name, instance, instance_name, data_format, data_value)
        gc.collect()
            
    
    