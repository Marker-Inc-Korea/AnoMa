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


# Set up logging
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





# METRICS_LIST = Configuration.metrics_list

# Influx DB basic information is taken from Configuration.
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
COLUMN_LIST = Configuration.column_list

ALL_TABLES = Configuration.all_tables
ALL_COLUMNS = Configuration.all_columns
print("ALL TABLES : {} , ALL_COLUMNS : {}".format(ALL_TABLES, ALL_COLUMNS))


# Imported except for the table at the head of the measurement called ANOMALY.
# If ALL_TABLES are true, all columns of each table are appended.
if ALL_TABLES:
    
    MEASUREMENTS_LIST = [measurement for measurement in db_client.get_list_measurements() if "ANOMALY" not in measurement['name'] ]

elif ALL_TABLES==False:
    MEASUREMENTS_LIST = []
    for k in db_client.get_list_measurements():
        if k['name'] in TABLE_LIST:
            MEASUREMENTS_LIST.append(k)

print(MEASUREMENTS_LIST)




# By extracting the detailed columns of each table
# Add to list
# If ALL_COLUMNS are true, all columns of each label are appended.
METRIC_LISTS = []

for measurement in MEASUREMENTS_LIST:
    metrics_list  = db_client.get_list_series(database = DATABASE_NAME, measurement = measurement['name'])
    for metric in metrics_list:
#         METRIC_LISTS.append(metric)
        result = {}
        
        
        if ALL_COLUMNS:
            metric = metric.split(',')
            result['measurement']  = metric[0]
            for item in metric[1:]:
                key, val = item.split('=',1)
                result[key] = val
            
            
        elif ALL_COLUMNS==False:
            col_list = COLUMN_LIST[metric[0]]
            for col in col_list:
                result['measurement']  = metric[0]
                result['label'] = col

            
      
        METRIC_LISTS.append(result)
# total list
print(METRIC_LISTS)
db_client.close()





# Command generation function for multi-threading
# Run the serving_model.py file
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
    )
    print(command)
    os.system(command)
    print("save done : {}_{}_{}".format(table_name,col_name,model_name))
    print('---- %s seconds ---- \n\n\n\n' % (time.time() - start_fac_time))
    
    

if __name__ == "__main__":
    shut_flag = False
    error_message = None
    try:
        with ThreadPoolExecutor(max_workers = len(METRIC_LISTS)) as excutor:
            
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