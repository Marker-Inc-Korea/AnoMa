"""doctsring for packages."""
import logging
from prometheus_api_client import Metric

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.from sklearn.ensemble import IsolationForest

# Set up logging
_LOGGER = logging.getLogger(__name__)


class MetricPredictor:
    """docstring for Predictor."""

    model_name = "isof"
    model_description = "Forecasted value from Isolation Forest model"
    model = None
    predicted_df = None
    metric = None

    def __init__(self, metric, rolling_data_window_size="10d"):
        """Initialize the Metric object."""
        self.metric = Metric(metric, rolling_data_window_size)
        
        
    def isof_model(self,n_estimators):
        clf = IsolationForest(n_etimators = 50, max_samples = 'auto')
        return clf

    def train(self, metric_data=None, prediction_duration=15):
        """Train the Prophet model and store the predictions in predicted_df."""
        prediction_freq = "1MIN"
        # convert incoming metric to Metric Object
        if metric_data:
            # because the rolling_data_window_size is set, this df should not bloat
            self.metric += Metric(metric_data)
            
        # normalising
        metric_values_np = self.metric.metric_values.values
        scaled_np_arr = self.sca

        # Don't really need to store the model, as prophet models are not retrainable
        # But storing it as an example for other models that can be retrained
        self.model = 
        
        if self.parameter_tuning:
            x, y = self.prepare_data(metric_values_np)
            lstm_cells = [2 ** i for i in range(5, 8)]
            dense_cells = [2 ** i for i in range(5, 8)]
            loss = np.inf
            lstm_cell_count = 0
            dense_cell_count = 0
            for lstm_cell_count_ in lstm_cells:
                for dense_cell_count_ in dense_cells:
                    model = self.get_model(lstm_cell_count_, dense_cell_count_)
                    model.compile(loss='mean_squared_error', optimizer='adam')
                    history = model.fit(x, y, epochs=50, batch_size=512, verbose=0,
                                        validation_split=self.validation_ratio)
                    val_loss = history.history['val_loss']
                    loss_ = min(val_loss)
                    if loss > loss_:
                        lstm_cell_count = lstm_cell_count_
                        dense_cell_count = dense_cell_count_
                        loss = loss_
            self.lstm_cell_count = lstm_cell_count
            self.dense_cell_count = dense_cell_count
            self.parameter_tuning = False
        
        
        
        
        
        
        
        
        
        
        
        self.model = Prophet(
            daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True
        )

        _LOGGER.info(
            "training data range: %s - %s", self.metric.start_time, self.metric.end_time
        )
        # _LOGGER.info("training data end time: %s", self.metric.end_time)
        _LOGGER.debug("begin training")

        self.model.fit(self.metric.metric_values)
        future = self.model.make_future_dataframe(
            periods=int(prediction_duration),
            freq=prediction_freq,
            include_history=False,
        )
        forecast = self.model.predict(future)
        forecast["timestamp"] = forecast["ds"]
        forecast = forecast[["timestamp", "yhat", "yhat_lower", "yhat_upper"]]
        forecast = forecast.set_index("timestamp")
        self.predicted_df = forecast
        _LOGGER.debug(forecast)

    def predict_value(self, prediction_datetime):
        """Return the predicted value of the metric for the prediction_datetime."""
        nearest_index = self.predicted_df.index.get_loc(
            prediction_datetime, method="nearest"
        )
        return self.predicted_df.iloc[[nearest_index]]