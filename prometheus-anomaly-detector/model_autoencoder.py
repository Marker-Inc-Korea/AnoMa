"""doctsring for packages."""
import logging
from prometheus_api_client import Metric
# from keras.models import Sequential, Model
# import keras.backend as K
# from keras.layers import Dense, Input, Conv1D, Dropout, Conv2DTranspose, Lambda
# from keras.layers import LSTM
from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow.keras 
# from keras.layers import Dense, Input, Conv1D, Dropout, Conv1DTranspose
# from keras.layers import LSTM

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Set up logging
_LOGGER = logging.getLogger(__name__)


class MetricPredictor:
    """docstring for Predictor."""

    model_name = "autoencoder"
    model_description = "Forecasted value from Auto-encoder model"
    model = None
    predicted_df = None
    metric = None

    def __init__(self, metric, rolling_data_window_size="10d", number_of_feature=10, validation_ratio=0.2,
                 parameter_tuning=True):
        """Initialize the Metric object."""
        self.metric = Metric(metric, rolling_data_window_size)

        self.number_of_features = number_of_feature
        self.scalar = MinMaxScaler(feature_range=(0, 1))
        self.parameter_tuning = parameter_tuning
        self.validation_ratio = validation_ratio

    def prepare_data(self, data):
        """Prepare the data for Auto-encoder."""
        train_x = np.array(data[:, 1])[np.newaxis, :].T

        for i in range(self.number_of_features):
            train_x = np.concatenate((train_x, np.roll(data[:, 1], -i)[np.newaxis, :].T), axis=1)

        train_x = train_x[:train_x.shape[0] - self.number_of_features, :self.number_of_features]

        train_yt = np.roll(data[:, 1], -self.number_of_features + 1)
        train_y = np.roll(data[:, 1], -self.number_of_features)
        train_y = train_y - train_yt
        train_y = train_y[:train_y.shape[0] - self.number_of_features]

        train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1 )
        return train_x, train_y
    
    
    def get_model(self, filter_size , kernel_size):
        """Build the model."""
#         filter_size = 16
#         kernel_size = 2
        inputs = layers.Input(shape = (self.number_of_features, 1))
        encoder = layers.Conv1D(filters = filter_size, kernel_size = kernel_size, 
                                padding = 'same', strides = 2, activation = 'relu')(inputs)
        encoder = layers.Dropout(rate = 0.2)(encoder)
        encoder = layers.Conv1D(filters = (filter_size/2), kernel_size = kernel_size,
                               padding = 'same', strides = 2, activation = 'relu')(encoder)
        
        decoder = layers.Conv1DTranspose(filters = (filter_size/2), kernel_size = kernel_size,
                               padding = 'same', strides = 2, activation = 'relu')(encoder)
        decoder = layers.Dropout(rate = 0.2)(decoder)
        decoder = layers.Conv1DTranspose(filters = filter_size, kernel_size = kernel_size,
                               padding = 'same', strides = 2, activation = 'relu')(decoder)
        outputs = layers.Conv1DTranspose(filters=1, kernel_size=kernel_size, padding = 'same')(decoder)
        
        model = keras.Model(inputs = inputs, outputs = outputs)
        print(model.summary())
        
#         model = Sequential()
#         model.add(LSTM(64, return_sequences=True, input_shape=(1, self.number_of_features)))
#         model.add(LSTM(lstm_cell_count))
#         model.add(Dense(dense_cell_count))
#         model.add(Dense(1))
#         print(model.summary())
        return model

    def train(self, metric_data=None, prediction_duration=15):
        """Train the model."""
        if metric_data:
            # because the rolling_data_window_size is set, this df should not bloat
            self.metric += Metric(metric_data)

        # normalising
        metric_values_np = self.metric.metric_values.values
        scaled_np_arr = self.scalar.fit_transform(metric_values_np[:, 1].reshape(-1, 1))
        metric_values_np[:, 1] = scaled_np_arr.flatten()

        if self.parameter_tuning:
            x, y = self.prepare_data(metric_values_np)
            filter_size = [2 ** i for i in range(2, 7)]
            kernel_size = [ i for i in range(2, 7)]
            loss = np.inf
            filter_size_count = 0
            kernel_size_count = 0
            for filter_size_count_ in filter_size:
                for kernel_size_count_ in kernel_size:
                    print(filter_size_count_, kernel_size_count_)
                    model = self.get_model(filter_size_count_, kernel_size_count_)
                    model.compile(loss='mean_squared_error', optimizer='adam')
                    print(x.shape, y.shape)
                    print(x.dtype, y.dtype)
                    print(x[:10], y[:10])
                    history = model.fit(x, y, epochs=50, batch_size=512, verbose=2,
                                        validation_split=self.validation_ratio)
                    val_loss = history.history['val_loss']
                    loss_ = min(val_loss)
                    if loss > loss_:
                        filter_size_count = filter_size_count_
                        kernel_size_count = kernel_size_count_
                        loss = loss_
            self.filter_size_count = filter_size_count
            self.kernel_size_count = kernel_size_count
            self.parameter_tuning = False

        model = self.get_model(self.filter_size_count, self.kernel_size_count)
        _LOGGER.info(
            "training data range: %s - %s", self.metric.start_time, self.metric.end_time
        )
        # _LOGGER.info("training data end time: %s", self.metric.end_time)
        _LOGGER.debug("begin training")
        data_x, data_y = self.prepare_data(metric_values_np)
        _LOGGER.debug(data_x.shape)
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(data_x, data_y, epochs=50, batch_size=512)
        data_test = metric_values_np[-self.number_of_features:, 1]
        forecast_values = []
        prev_value = data_test[-1]
        for i in range(int(prediction_duration)):
            prediction = model.predict(data_test.reshape(1,self.number_of_features,  1 )).flatten()[0]
            curr_pred_value = data_test[-1] + prediction
            scaled_final_value = self.scalar.inverse_transform(curr_pred_value.reshape(1, -1)).flatten()[0]
            forecast_values.append(scaled_final_value)
            data_test = np.roll(data_test, -1)
            data_test[-1] = curr_pred_value
            prev_value = data_test[-1]

        dataframe_cols = {"yhat": np.array(forecast_values)}

        upper_bound = np.array(
            [
                (
                        forecast_values[i] + (np.std(forecast_values[:i]) * 2)
                )
                for i in range(len(forecast_values))
            ]
        )
        upper_bound[0] = np.mean(
            forecast_values[0]
        )  # to account for no std of a single value
        lower_bound = np.array(
            [
                (
                        forecast_values[i] - (np.std(forecast_values[:i]) * 2)
                )
                for i in range(len(forecast_values))
            ]
        )
        lower_bound[0] = np.mean(
            forecast_values[0]
        )  # to account for no std of a single value
        dataframe_cols["yhat_upper"] = upper_bound
        dataframe_cols["yhat_lower"] = lower_bound

        data = self.metric.metric_values
        maximum_time = max(data["ds"])
        dataframe_cols["timestamp"] = pd.date_range(
            maximum_time, periods=len(forecast_values), freq="min"
        )

        forecast = pd.DataFrame(data=dataframe_cols)
        forecast = forecast.set_index("timestamp")

        self.predicted_df = forecast
        _LOGGER.debug(forecast)

    def predict_value(self, prediction_datetime):
        """Return the predicted value of the metric for the prediction_datetime."""
        nearest_index = self.predicted_df.index.get_loc(
            prediction_datetime, method="nearest"
        )
        return self.predicted_df.iloc[[nearest_index]]