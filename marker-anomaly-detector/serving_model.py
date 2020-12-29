import time
import os
import logging
from datetime import datetime, timedelta
import argparse
import importlib
from influxdb import InfluxDBClient
import pandas as pd
import numpy as np
from configuration import Configuration

import schedule

import warnings
warnings.filterwarnings('ignore')



# Set up logging
_LOGGER = logging.getLogger(__name__)

# Set up tarin model
parser = argparse.ArgumentParser(description = "Argparser_train_model")
parser.add_argument('--table_name',type = str,
                   help = 'select model name')
parser.add_argument('--col_name',type = str,
                   help = 'select model name')
parser.add_argument('--model_name',type = str,
                   help = 'select model name')
parser.add_argument('--rolling_size',type = str,
                   help = 'select model name')
parser.add_argument('--retrain_interval',type = str,
                   help = 'select model name')
parser.add_argument('--number_of_feature',type = int,
                   help = 'select model name')
parser.add_argument('--model_dir',type = str,
                   help = 'select model name')
args = parser.parse_args()
print(args.model_name)

model = importlib.import_module(args.model_name)

# Influx DB basic information is taken from Configuration.
DATABASE_URL = Configuration.influxdb_url
DATABASE_PORT = Configuration.influxdb_port
DATABASE_NAME = Configuration.influxdb_name
DATABSAE_USERNAME = Configuration.influxdb_username
DATABSAE_PASSWORD = Configuration.influxdb_password
# DATABASE_POLICY = 'autogen'


db_client = InfluxDBClient(DATABASE_URL, DATABASE_PORT, DATABSAE_USERNAME,DATABSAE_PASSWORD,  DATABASE_NAME)




start_time = None


# It tracks database updates in real time.
# Also, according to the retrain_interval variable, 
# the model is updated according to the model update time.
# The final prediction results anomaly, yraw, yhat, yhat_lower, and yhat_upper are saved in Influxdb.
# When saved, the name of the table is created in the form "ANOMALY_{table_name}_{model}".
def serving_model(model=None, rolling_size = '3d', retrain_interval = 3):
    
    global start_time, end_time
    
    # train model setting
    model = model.MetricPredictor(table_name = args.table_name, 
                              col_name = args.col_name, 
                              rolling_data_window_size = Configuration.rolling_training_window_size,
                              model_dir = args.model_dir
                                 )
    
    model.reload_model(initialize_load = True)
    last_claues = """SELECT last("value") FROM "prometheus"."autogen"."{table_name}" WHERE "label"='{col_name}' """.format(
        table_name = args.table_name,
        col_name = args.col_name
    )
    print(last_claues)
    retrain_time = datetime.now() + timedelta(minutes=(retrain_interval) + 0.1)
    current_time = datetime.now()
    
    while True:
        time.sleep(0.005)
        last_df = pd.DataFrame(db_client.query(last_claues).get_points())
        last_time = datetime.strptime( last_df['time'].values[0], '%Y-%m-%dT%H:%M:%S.%fZ')
        
        
        # retrain every retrain interval
        if current_time >= retrain_time:
            model.reload_model(initialize_load=False)
            retrain_time = datetime.now() + timedelta(minutes=(retrain_interval) + 0.1)
            _LOGGER.info(
                "Will reload model every %s minutes", 3
            )
            continue
            
        # continue - not yet newest value insert to database
        if current_time>=last_time:
            continue
        
        _LOGGER.info('VALUE IN time : {} - value : {} '.format(last_time, last_df['last'].values))
        
        prediction = model.predict_value(last_df['last'].values)
        
        
        yhat = prediction[0]
        yhat_lower = prediction[1]
        yhat_upper = prediction[2]
        anomaly = prediction[3]
        
        
        _LOGGER.info('PREDICT  time : {time} -- yvalue : {yvalue}  yhat : {yhat}  yhat_lower : {yhat_lower}   yhat_upper : {yhat_upper}'.format(
            time = last_time,
            yvalue = last_df['last'].values,
            yhat = yhat,
            yhat_lower = yhat_lower,
            yhat_upper = yhat_upper,
            
        ))

        
        
        _LOGGER.info("Anomaly - time : {} value : {}".format(last_time, anomaly))
        
        
        json_body = [
            {
                "measurement" : "ANOMALY_{table_name}_{model}".format(model = args.model_name,
                                                                      table_name = args.table_name),
                "label" : args.col_name,
                "time" : last_time,
                "fields" : {
                    "anomaly" :  anomaly,
                    "yraw" : last_df['last'].values,
                    "yhat" : yhat,
                    "yhat_lower" : yhat_lower,
                    "yhat_upper" : yhat_upper
                }
            }
        ]
        
    
        db_client.write_points(json_body ,time_precision ='u')
        _LOGGER.info('SEND VALUE : {time}'.format(time = last_time))
        
        
        current_time = datetime.now()

        
if __name__ == "__main__":
    
    serving_model(model , Configuration.retraining_interval_minutes)
    