"""docstring for packages."""
import time
import os
import logging
from datetime import datetime
from configuration import Configuration
import schedule

import argparse
import importlib
from influxdb import InfluxDBClient


from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import sys
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

workers = os.cpu_count()

_LOGGER = logging.getLogger(__name__)

# Set up model
parser = argparse.ArgumentParser(description = "Argparser_app")
parser.add_argument('--model_name',type = str,
                   help = 'select model name',
                   default = 'model')
args = parser.parse_args()
model_name = args.model_name
print(model_name)
print(os.path.join(os.getcwd(),args.model_name))


# Set up logging


# METRICS_LIST = Configuration.metrics_list


DATABASE_URL = Configuration.influxdb_url
DATABASE_PORT = Configuration.influxdb_port
DATABASE_NAME = Configuration.influxdb_name
DATABSAE_USERNAME = Configuration.influxdb_username
DATABSAE_PASSWORD = Configuration.influxdb_password

ROLLING_WINDOW_SIZE = Configuration.rolling_training_window_size
RETRAINING_INTERVAL = Configuration.retraining_interval_minutes
MODEL_DIR = Configuration.model_dir


db_client = InfluxDBClient(DATABASE_URL, DATABASE_PORT, DATABSAE_USERNAME,DATABSAE_PASSWORD,  DATABASE_NAME)

TABLE_LIST = Configuration.table_list

ALL_METRICS = Configuration.all_metrics
print(ALL_METRICS)
METRIC_LISTS = []
#     MEASUREMENT_LIST = db_client.get_list_measurements()
if ALL_METRICS:
    MEASUREMENTS_LIST = db_client.get_list_measurements()
    print('AAAA')
elif ALL_METRICS==False:
    MEASUREMENTS_LIST = []
    for k in db_client.get_list_measurements():
        if k['name'] in TABLE_LIST:
            MEASUREMENTS_LIST.append(k)

print(MEASUREMENTS_LIST)
# quit()
# METRICS_LIST = db_client.get_list_series(database=DATABASE_NAME)




for measurement in MEASUREMENTS_LIST:
    metrics_list  = db_client.get_list_series(database = DATABASE_NAME, measurement = measurement['name'])
    for metric in metrics_list:
#         METRIC_LISTS.append(metric)
        result = {}
        metric = metric.split(',')
        result['measurement']  = metric[0]
        for item in metric[1:]:
            key, val = item.split('=',1)
            result[key] = val
        METRIC_LISTS.append(result)
# total list
print(METRIC_LISTS)

db_client.close()


## PO model serving multi        
def command_gen(table_name, col_name, model_name, rolling_size , retrain_interval, model_dir):
    time.sleep(1)
    print('Start {}_{}_{}'.format(table_name,col_name,model_name))
    start_fac_time = time.time()
    
    command = "python serving_model.py --table_name {table_name} --col_name {col_name} --model_name {model_name} --rolling_size {rolling_size} --retrain_interval {retrain_interval} --model_dir {model_dir}".format(
        model_name = model_name,
        table_name = table_name, 
        col_name = col_name , 
        rolling_size = rolling_size,
        retrain_interval = retrain_interval,
        model_dir = model_dir
#     input_dir = input_dir,
#     output_dir = output_dir,
#     model_name = model_name,
#     fac_name = fac_name
    )
    print(command)
    os.system(command)
    print("save done : {}_{}_{}".format(table_name,col_name,model_name))
    print('---- %s seconds ---- \n\n\n\n' % (time.time() - start_fac_time))
    
    

if __name__ == "__main__":
    shut_flag = False
    error_message = None
    try:
        with ThreadPoolExecutor(max_workers = workers) as excutor:
            
            try:
                for metric_list in METRIC_LISTS:
                    table_name = metric_list['measurement']
                    col_name = metric_list['label']


                    
                    time.sleep(1)
                    print("Table :  {table_name}, Col : {col_name}, rolling_size : {rolling_size}, retrain_interval : {retrain_interval}, Model : {model_name} model , Model_dir : {model_dir}  online learning".format(
                        table_name = table_name, 
                        col_name = col_name,
                        rolling_size = ROLLING_WINDOW_SIZE,
                        retrain_interval = RETRAINING_INTERVAL,
                        model_name = model_name,
                        model_dir = MODEL_DIR))
                    
                    excutor.submit(command_gen, table_name, col_name, model_name, ROLLING_WINDOW_SIZE, RETRAINING_INTERVAL, MODEL_DIR)
            except RuntimError as e:
                print(e)
                shut_flag = True
            except Exception as e:
                print(e)
                shut_flag = True
            except KeyboardInterrupt:
                print('Interrupted')
                shut_flag = True
            if shut_flag==True:
                print('shutdown - model learning')
                excutor.shutdown(wait=True)
        print('done')
    
    
    except KeyboardInterrupt:
        print('Interrupted')