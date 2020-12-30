"""doctsring for packages."""
import logging
# from prometheus_api_client import Metric

import json
from tqdm import tqdm
import os
import tensorflow as tf
# import pickle
import joblib

from models.lstm_vae import LSTM_Var_Autoencoder

import numpy as np
np.random.seed(2020)
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pathlib import Path

import tempfile



# Set up logging
_LOGGER = logging.getLogger(__name__)

# graph = tf.get_default_graph()



class MetricPredictor:
    """docstring for Predictor."""
    
    
    
    model_name = "lstmvae"
    model_description = "Forecasted value from lstmvae model"
    model = None
    predicted_df = None
    metric = None
#     raw_roll = None
    
    

    def __init__(self, table_name, col_name,model_dir, rolling_data_window_size="10d"  ,number_of_feature=10):
        """Initialize the Metric object."""
#         self.metric = Metric(metric, rolling_data_window_size)

        self.number_of_features = number_of_feature
        self.scalar = MinMaxScaler(feature_range=(-1, 1))
#         self.model = None
#         self.parameter_tuning = parameter_tuning
#         self.validation_ratio = validation_ratio
#         self.mean = None
#         self.std = None
        self.table_name = table_name
        self.col_name = col_name
        self.raw_roll = np.zeros((1,self.number_of_features,1))
        self.x_dim = 30
        self.z_dim = 10
        self.model_dir = model_dir

#     def prepare_data(self, data):
#         """Prepare the data for LSTM."""
#         train_x = np.array(data[:, 1])[np.newaxis, :].T
# #         print(train_x.shape)
#         for i in range(self.number_of_features):
#             train_x = np.concatenate((train_x, np.roll(data[:, 1], -i)[np.newaxis, :].T), axis=1)
# #         print(train_x.shape)
#         train_x = train_x[:train_x.shape[0] - self.number_of_features, :self.number_of_features]

#         train_yt = np.roll(data[:, 1], -self.number_of_features + 1)
#         train_y = np.roll(data[:, 1], -self.number_of_features)
#         train_y = train_y - train_yt
#         train_y = train_y[:train_y.shape[0] - self.number_of_features]

#         train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])
        
#         return train_x, train_y

    def unroll_auto(self, data, sequence_length=24):
        result = []
        for index in range(len(data) - sequence_length + 1):
            result.append(data[index: index + sequence_length])
        return np.asarray(result)

    def get_model(self, lstm_dim = 30, z_dim = 10, predict_=False):
        """Build the model."""
        
        
        vae  = LSTM_Var_Autoencoder(intermediate_dim= lstm_dim, z_dim=z_dim, n_dim=1,  stateful = True, predict_=predict_)

        return vae
    
    def reload_model(self, initialize_load = True):
        if initialize_load:
            self.model_dir = os.getcwd()+'/'+self.model_dir + '/{table_name}_{col_name}/{model_name}'.format(table_name = self.table_name, col_name = self.col_name, model_name = self.model_name)
            print(self.model_dir)
            
        
        self.model =  self.get_model(self.x_dim, self.z_dim, True)
        self.model.model_dir = self.model_dir
        self.model.load_model()
        scalar_file_name = self.model_dir + '/' + 'scalar.pkl'
        self.scalar = joblib.load(scalar_file_name)
        _LOGGER.info("RESTORING MODEL RETURN START")
#         model.load_model(self.model_dir)
#         _LOGGER.info("RESTORING MODEL RETURN")
#         return model
    
    def get_variable(self):
        _LOGGER.info("RETRUN MODEL VARIABLE")
        variables = {"model_variable" : [self.x_dim, self.z_dim],
                    "model_savedir" : self.model_dir}
        return variables

    def train(self, metric_data=None, prediction_duration=15):
        """Train the model."""
        
        prediction_duration = prediction_duration*60
        
        # normalising
        self.metric = metric_data
        _LOGGER.info(self.metric.shape)
#         quit()
        timestamps = self.metric.iloc[:,0]
        metric_values_np = self.metric.iloc[:,1].values
#         self.scalar = self.scalar.fit(metric_values_np.reshape(-1,1))
        scaled_np_arr = self.scalar.fit_transform(metric_values_np.reshape(-1,1))
        
        
        
#         if metric_data:
#             # because the rolling_data_window_size is set, this df should not bloat
#             self.metric += Metric(metric_data)
            
#         # normalising
#         timestamps = self.metric.metric_values.ds
#         metric_values_np = self.metric.metric_values.values
#         scaled_np_arr = self.scalar.fit_transform(metric_values_np[:,1].reshape(-1,1))
        

        model = self.get_model(self.x_dim, self.z_dim)
        _LOGGER.info(
            "training data range: %s - %s", timestamps.iloc[0], timestamps.iloc[-1]
        )
        _LOGGER.info("begin training")
#         quit()
        # change to seq data format
        x_train_rolling = self.unroll_auto(scaled_np_arr, self.number_of_features)
        
#         _LOGGER.debug(x_train_rolling.shape)
        _LOGGER.info(x_train_rolling.shape)
        
        model.fit(x_train_rolling, learning_rate=0.001, batch_size = 100, 
                num_epochs = 100, opt = tf.train.AdamOptimizer, REG_LAMBDA = 0.01,
                grad_clip_norm=10, optimizer_params=None, verbose = True)
#         quit()
        
        self.model_dir = os.getcwd()+'/'+self.model_dir + '/{table_name}_{col_name}/{model_name}'.format(table_name = self.table_name, col_name = self.col_name, model_name = self.model_name)
        print(self.model_dir)
#         quit()
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
#         quit()
#         self.model_dir = '/PAD/MODEL/'
        
        
        model.save_model(self.model_dir)
        
        
        scalar_file_name = self.model_dir + '/' + 'scalar.pkl'
#         pickle.dump(scaler, open('scaler.scl','wb'))
#         self.scalar = pickle.load(open(scalar_file_name, 'rb'))
        joblib.dump(self.scalar, scalar_file_name)
#         pickle.dump(self.scalar, open(scalar_file_name,'wb'))
        _LOGGER.info("Save model - {} ".format(self.model_dir))
        
        _LOGGER.info("Training end")

    
    def predict_value(self, predict_value):
        """Return the predicted value of the metric for the prediction_datetime."""
#         model.load_model(self.model_dir)
#         _LOGGER.info("RESTORING MODEL RETURN")
        
#         
#         global graph
#         graph = tf.get_default_graph()

        
        self.raw_roll = np.roll(self.raw_roll, -1)
        _LOGGER.info("Rolling BEFORE")
        predict_value_pp = self.scalar.transform(predict_value.reshape(-1,1))
        _LOGGER.info("Scalar transform")
        self.raw_roll[:,-1, : ] = predict_value_pp
#         print(self.raw_roll)
#         _LOGGER.info
        _LOGGER.info(self.raw_roll.shape)
#         graph = tf.get_default_graph()
#         with model.graph.as_default():
            
        x_reconstructed, recons_error = self.model.reconstruct(self.raw_roll, get_error = True)
        _LOGGER.info("Model reconstruction")
        dataframe_cols = {"yhat": np.array(self.scalar.inverse_transform(x_reconstructed[:,-1, : ]))}
        upper_bound = np.array(
            [
                (
                    self.scalar.inverse_transform(x_reconstructed[:, -1,:]) + (np.std(self.scalar.inverse_transform(x_reconstructed[0, :-1])) * 4 )
                ) 
            ]
        ).flatten()
        
        
        lower_bound = np.array(
            [
                (
                    self.scalar.inverse_transform(x_reconstructed[:, -1,:]) - (np.std(self.scalar.inverse_transform(x_reconstructed[0, :-1])) * 4 )
                ) 
            ]
        ).flatten()
        
        
        
        
        dataframe_cols["yhat_upper"] = upper_bound
        dataframe_cols["yhat_lower"] = lower_bound
#         _LOGGER.info("YHAT : {} UPPER : {}  LOWER : {}".format(dataframe_cols["yhat"], 
#                                                                dataframe_cols["yhat_upper"],
#                                                                dataframe_cols["yhat_lower"]
#                                                               )
#                     )
        
        # prophet model(origin value base)
#         anomaly = 1
#         if (
#                 last_df['last'].values < dataframe_cols["yhat_upper"]
#             ) and (
#                 last_df['last'].values > dataframe_cols["yhat_lower"]
#             ):
#                 anomaly = 0
        
    
        # lstm_vae model(reconstruct value base)
        anomaly = 1
        if (
                dataframe_cols["yhat"] < dataframe_cols["yhat_upper"]
            ) and (
                dataframe_cols["yhat"] > dataframe_cols["yhat_lower"]
            ):
                anomaly = 0
                
                
        
        
        return (dataframe_cols["yhat"], dataframe_cols["yhat_lower"], dataframe_cols["yhat_upper"], anomaly)
        
        
        
        
        
        
        
        
        
        
#         forecast = pd.DataFrame(data = dataframe_cols)
        
#         print(dataframe_cols)
        
        
        
        
        
#         nearest_index = self.predicted_df.index.get_loc(
#             prediction_datetime, method="nearest"
#         )
#         return dataframe_cols
    
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#         labels = np.zeros_like(values, dtype=np.int32)
#         timestamps = pd.DatetimeIndex(timestamps).astype(np.int64)
        
        
#         train_values, mean, std = standardize_kpi(
#             train_values, excludes=np.logical_or(train_labels, train_missing))
#         self.mean = mean
#         self.std = std
        
        
#         model_h_for_p_x, model_h_for_q_z = get_model()
#         model_dounut = Donut(model_h_for_p_x, model_h_for_q_z, x_dims=self.x_dims, z_dims=self.z_dim)
#         trainer = DonutTrainer(model = model_donut, max_epoch = 100)
#         predictor = DonutPredictor(model_donut)

        
        
        
        
        
#         _LOGGER.info(
#             "training data range: %s - %s", self.metric.start_time, self.metric.end_time
#         )
#         _LOGGER.debug("begin training")
#         _LOGGER.debug(values.shape)
        
#         sess = tf.InteractiveSession()
#         trainer.fit(values = train_values,
#         labels = train_labels,
#         missing = train_missing,
#         mean = mean,
#         std= std,
#         valid_portion = self.validation_ratio            
#                    )
        
#         model = predictor
        
        
        
        
        
        
        
        
        
        
        
        
        
       
        
        
        
        
        
#         t_idxs = pd.DatetimeIndex(abnormal.index).astype(np.int64)
        
        
#         print(self.metric)
#         print(self.metric.metric_values.values)
#         print(pd.DatetimeIndex(self.metric.metric_values.ds).astype(np.int64))
#         aa
        
        
        
        
#         metric_values_np = self.metric.metric_values.values
            
            
            
    

#         # normalising
#         metric_values_np = self.metric.metric_values.values
#         scaled_np_arr = self.scalar.fit_transform(metric_values_np[:, 1].reshape(-1, 1))
#         metric_values_np[:, 1] = scaled_np_arr.flatten()

#         if self.parameter_tuning:
#             x, y = self.prepare_data(metric_values_np)
#             lstm_cells = [2 ** i for i in range(5, 8)]
#             dense_cells = [2 ** i for i in range(5, 8)]
#             loss = np.inf
#             lstm_cell_count = 0
#             dense_cell_count = 0
#             for lstm_cell_count_ in lstm_cells:
#                 for dense_cell_count_ in dense_cells:
#                     model = self.get_model(lstm_cell_count_, dense_cell_count_)
#                     model.compile(loss='mean_squared_error', optimizer='adam')
#                     history = model.fit(x, y, epochs=50, batch_size=512, verbose=0,
#                                         validation_split=self.validation_ratio)
#                     val_loss = history.history['val_loss']
#                     loss_ = min(val_loss)
#                     if loss > loss_:
#                         lstm_cell_count = lstm_cell_count_
#                         dense_cell_count = dense_cell_count_
#                         loss = loss_
#             self.lstm_cell_count = lstm_cell_count
#             self.dense_cell_count = dense_cell_count
#             self.parameter_tuning = False

#         model = self.get_model(self.lstm_cell_count, self.dense_cell_count)
#         _LOGGER.info(
#             "training data range: %s - %s", self.metric.start_time, self.metric.end_time
#         )
#         # _LOGGER.info("training data end time: %s", self.metric.end_time)
#         _LOGGER.debug("begin training")
#         data_x, data_y = self.prepare_data(metric_values_np)
#         _LOGGER.debug(data_x.shape)
#         model.compile(loss='mean_squared_error', optimizer='adam')
#         model.fit(data_x, data_y, epochs=50, batch_size=512)
#         data_test = metric_values_np[-self.number_of_features:, 1]
#         forecast_values = []
#         prev_value = data_test[-1]
#         for i in range(int(prediction_duration)):
#             prediction = model.predict(data_test.reshape(1, 1, self.number_of_features)).flatten()[0]
#             curr_pred_value = data_test[-1] + prediction
#             scaled_final_value = self.scalar.inverse_transform(curr_pred_value.reshape(1, -1)).flatten()[0]
#             forecast_values.append(scaled_final_value)
#             data_test = np.roll(data_test, -1)
#             data_test[-1] = curr_pred_value
#             prev_value = data_test[-1]

#         dataframe_cols = {"yhat": np.array(forecast_values)}

#         upper_bound = np.array(
#             [
#                 (
#                         forecast_values[i] + (np.std(forecast_values[:i]) * 2)
#                 )
#                 for i in range(len(forecast_values))
#             ]
#         )
#         upper_bound[0] = np.mean(
#             forecast_values[0]
#         )  # to account for no std of a single value
#         lower_bound = np.array(
#             [
#                 (
#                         forecast_values[i] - (np.std(forecast_values[:i]) * 2)
#                 )
#                 for i in range(len(forecast_values))
#             ]
#         )
#         lower_bound[0] = np.mean(
#             forecast_values[0]
#         )  # to account for no std of a single value
#         dataframe_cols["yhat_upper"] = upper_bound
#         dataframe_cols["yhat_lower"] = lower_bound

#         data = self.metric.metric_values
#         maximum_time = max(data["ds"])
#         dataframe_cols["timestamp"] = pd.date_range(
#             maximum_time, periods=len(forecast_values), freq="min"
#         )

#         forecast = pd.DataFrame(data=dataframe_cols)
#         forecast = forecast.set_index("timestamp")

#         self.predicted_df = forecast
#         _LOGGER.debug(forecast)

#     def predict_value(self, prediction_datetime):
#         """Return the predicted value of the metric for the prediction_datetime."""
#         nearest_index = self.predicted_df.index.get_loc(
#             prediction_datetime, method="nearest"
#         )
#         return self.predicted_df.iloc[[nearest_index]]
