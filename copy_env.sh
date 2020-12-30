#!/bin/sh
BASE_DIR=`pwd -P`

# cp env_template $BASE_DIR'/.env'

PUSHGATE_DIR=$BASE_DIR'/pushgateway/'
PROMETHEUS_DIR=$BASE_DIR'/prometheus/'
PAD_DIR=$BASE_DIR'/prometheus-anomaly-detector/'
MLFLOW_DIR=$BASE_DIR'/mlflow/'
GRAFANADB_DIR=$BASE_DIR'/grafana_db/'

cp .env $PUSHGATE_DIR'.env'
cp .env $PROMETHEUS_DIR'.env'
cp .env $PAD_DIR'.env'
cp .env $PUSHGATE_DIR'.env'
cp .env $MLFLOW_DIR'.env'
cp .env $GRAFANADB_DIR'.env'
# cp .env $PROMETHEUS_DIR'application.properties'