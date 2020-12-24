"""docstring for packages."""
import time
import os
import logging
from datetime import datetime
# from multiprocessing import Process, Queue
from multiprocessing import Process, Manager
# from queue import Empty as EmptyQueueException
# import tornado.ioloop
# import tornado.web
# from prometheus_client import Gauge, generate_latest, REGISTRY
# from prometheus_api_client import PrometheusConnect, Metric
from configuration import Configuration
# import tensorflow as tf
# import model
# import model_lstm as model
# import model_fourier as model
import schedule

import argparse
import importlib



# import tensorflow as tf 
# tf.executing_eagerly()




# Set up model
parser = argparse.ArgumentParser(description = "Argparser_app")
parser.add_argument('--model_name',type = str,
                   help = 'select model name',
                   default = 'model')
args = parser.parse_args()
print(args.model_name)
# print(os.path.join(os.getcwd(),args.model_name))


# spec = importlib.util.spec_from_file_location("module.name", args.model_name)
# model = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(model)
model = importlib.import_module(args.model_name)







# Set up logging
_LOGGER = logging.getLogger(__name__)

METRICS_LIST = Configuration.metrics_list

# list of ModelPredictor Objects shared between processes
PREDICTOR_MODEL_LIST = list()


pc = PrometheusConnect(
    url=Configuration.prometheus_url,
    headers=Configuration.prom_connect_headers,
    disable_ssl=True,
)

for metric in METRICS_LIST:
    # Initialize a predictor for all metrics first
    metric_init = pc.get_current_metric_value(metric_name=metric)

    for unique_metric in metric_init:
        PREDICTOR_MODEL_LIST.append(
            model.MetricPredictor(
                unique_metric,
                rolling_data_window_size=Configuration.rolling_training_window_size,
            )
        )

# A gauge set for the predicted values
GAUGE_DICT = dict()
for predictor in PREDICTOR_MODEL_LIST:
    unique_metric = predictor.metric
    label_list = list(unique_metric.label_config.keys())
    label_list.append("value_type")
    if unique_metric.metric_name not in GAUGE_DICT:
        GAUGE_DICT[unique_metric.metric_name] = Gauge(
            unique_metric.metric_name + "_" + predictor.model_name,
            predictor.model_description,
            label_list,
        )

        
class MainHandler(tornado.web.RequestHandler):
    """Tornado web request handler."""

    def initialize(self, data_queue):
        """Check if new predicted values are available in the queue before the get request."""
        _LOGGER.info(data_queue)
        _LOGGER.info('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
        try:
            model_list = data_queue.get_nowait()
            _LOGGER.info(model_list)
            _LOGGER.info('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
            self.settings["model_list"] = model_list
        except EmptyQueueException:
            pass

    async def get(self):
        """Fetch and publish metric values asynchronously."""
        # update metric value on every request and publish the metric
        for predictor_model in self.settings["model_list"]:
            # get the current metric value so that it can be compared with the
            # predicted values
            current_metric_value = Metric(
                pc.get_current_metric_value(
                    metric_name=predictor_model.metric.metric_name,
                    label_config=predictor_model.metric.label_config,
                )[0]
            )

            metric_name = predictor_model.metric.metric_name
#             prediction = predictor_model.predict_value(datetime.now())
            
            prediction = predictor_model.predict_value(current_metric_value.metric_values["y"][0])

            # Check for all the columns available in the prediction
            # and publish the values for each of them
            for column_name in list(prediction.columns):
                GAUGE_DICT[metric_name].labels(
                    **predictor_model.metric.label_config, value_type=column_name
                ).set(prediction[column_name][0])

            # Calculate for an anomaly (can be different for different models)
#             _LOGGER.info(current_metric_value.metric_values["y"])
#             _LOGGER.info('BBBBB')
#             _LOGGER.info(prediction)
#             _LOGGER.info('AAAAAA')
            anomaly = 1
            if (
                current_metric_value.metric_values["y"][0] < prediction["yhat_upper"][0]
            ) and (
                current_metric_value.metric_values["y"][0] > prediction["yhat_lower"][0]
            ):
                anomaly = 0
            _LOGGER.info("Anomaly - value : {}".format(anomaly))
            # create a new time series that has value_type=anomaly
            # this value is 1 if an anomaly is found 0 if not
            GAUGE_DICT[metric_name].labels(
                **predictor_model.metric.label_config, value_type="anomaly"
            ).set(anomaly)

        self.write(generate_latest(REGISTRY).decode("utf-8"))
        self.set_header("Content-Type", "text; charset=utf-8")


def make_app(data_queue):
    """Initialize the tornado web app."""
    _LOGGER.info("Initializing Tornado Web App")
    
#     aaaaaaa = tornado.web.Application(
#         [
#             (r"/metrics", MainHandler, dict(data_queue=data_queue)),
#             (r"/", MainHandler, dict(data_queue=data_queue)),
#         ]
#     )
#     _LOGGER.info("Initializing Tornado Web AppAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
#     return aaaaaaa
    return tornado.web.Application(
        [
            (r"/metrics", MainHandler, dict(data_queue=data_queue)),
            (r"/", MainHandler, dict(data_queue=data_queue)),
        ]
    )


def train_model(initial_run=False, data_queue=None):
    """Train the machine learning model."""
    for predictor_model in PREDICTOR_MODEL_LIST:
        metric_to_predict = predictor_model.metric
        data_start_time = datetime.now() - Configuration.metric_chunk_size
        if initial_run:
            data_start_time = (
                datetime.now() - Configuration.rolling_training_window_size
            )

        # Download new metric data from prometheus
        new_metric_data = pc.get_metric_range_data(
            metric_name=metric_to_predict.metric_name,
            label_config=metric_to_predict.label_config,
            start_time=data_start_time,
            end_time=datetime.now(),
        )[0]

        # Train the new model
        start_time = datetime.now()
        predictor_model.train(
            new_metric_data, Configuration.retraining_interval_minutes
        )
        _LOGGER.info(
            "Total Training time taken = %s, for metric: %s %s",
            str(datetime.now() - start_time),
            metric_to_predict.metric_name,
            metric_to_predict.label_config,
        )
    _LOGGER.info(PREDICTOR_MODEL_LIST)
    _LOGGER.info('BBBBBBBBBBBBBBBBBBBBBBBBB')
    data_queue.put(PREDICTOR_MODEL_LIST)
    _LOGGER.info(data_queue)


if __name__ == "__main__":
    # Queue to share data between the tornado server and the model training
#     predicted_model_queue = Queue()
    manager = Manager()
#     manager.start()
    predicted_model_queue = manager.Queue()

    # Initial run to generate metrics, before they are exposed
    train_model(initial_run=True, data_queue=predicted_model_queue)

    # Set up the tornado web app
    app = make_app(predicted_model_queue)
    app.listen(8080)
    _LOGGER.info('CCCCCCCCCCCCCCCCCCCCC')
    server_process = Process(target=tornado.ioloop.IOLoop.instance().start)
    _LOGGER.info('DDDDDDDDDDDDDDDDDDDDDDDDDDD')
    # Start up the server to expose the metrics.
    server_process.start()
    _LOGGER.info('EEEEEEEEEEEEEEEEEEEEEEEE')
    # Schedule the model training
    schedule.every(Configuration.retraining_interval_minutes).minutes.do(
        train_model, initial_run=False, data_queue=predicted_model_queue
    )
    _LOGGER.info(
        "Will retrain model every %s minutes", Configuration.retraining_interval_minutes
    )

    while True:
        schedule.run_pending()
        time.sleep(1)

    # join the server process in case the main process ends
    server_process.join()