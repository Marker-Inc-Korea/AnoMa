import time
import os
import logging
from datetime import datetime
import argparse
import importlib
from influxdb import InfluxDBClient
import pandas as pd
import numpy as np
from configuration import Configuration

import schedule



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
# print(os.path.join(os.getcwd(),args.model_name))


# spec = importlib.util.spec_from_file_location("module.name", args.model_name)
# model = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(model)

# model_import 


model = importlib.import_module(args.model_name)


# influx db client setting 
DATABASE_URL = Configuration.influxdb_url
DATABASE_PORT = Configuration.influxdb_port
DATABASE_NAME = Configuration.influxdb_name
DATABSAE_USERNAME = Configuration.influxdb_username
DATABSAE_PASSWORD = Configuration.influxdb_password




db_client = InfluxDBClient(DATABASE_URL, DATABASE_PORT, DATABSAE_USERNAME,DATABSAE_PASSWORD,  DATABASE_NAME)


def train_model(model=None, initial_run = False, rolling_size = '3d'):
    
    # train model setting
    model = model.MetricPredictor(table_name = args.table_name, 
                              col_name = args.col_name, 
                              rolling_data_window_size = Configuration.rolling_training_window_size,
                              model_dir = args.model_dir,
                                 number_of_feature = args.number_of_feature)
    

    data_end_time = datetime.now()
    data_start_time = data_end_time  - Configuration.metric_chunk_size
    if initial_run:
        data_start_time = (
            data_end_time - Configuration.rolling_training_window_size
        )
        
        
    # Download new metric datat from influx db
    select_clause = """SELECT "value" FROM "prometheus"."autogen"."{table_name}" WHERE time > '{start_time}' AND time < '{end_time}' AND "label"='{col_name}'""".format(
    table_name = args.table_name,
    col_name = args.col_name,
    start_time = data_start_time,
    end_time = data_end_time
)
    new_metric_data = pd.DataFrame(db_client.query(select_clause).get_points())
    
    print(new_metric_data)
#     quit()
    # Train the new model
    start_time = datetime.now()
    model.train(new_metric_data, Configuration.retraining_interval_minutes)
    _LOGGER.info(
            "Total Training time taken = %s, for table: %s %s",
            str(datetime.now() - start_time),
            model.table_name,
            model.col_name,
        )
    
#     quit()


if __name__ == "__main__":
    
    train_model(model , initial_run=True)
    
    # Schedule the model training
    schedule.every(Configuration.retraining_interval_minutes).minutes.do(
        train_model, model= model, initial_run=False
    )
    _LOGGER.info(
        "Will retrain model every %s minutes", Configuration.retraining_interval_minutes
    )

    while True:
        schedule.run_pending()
        time.sleep(1)
    
    