## Run

Creadte DB folder
```
mkdir mlflow_docker
cd mlflow_docker
mkdir mlflow_server
mkdir mysql
```

To launch a local MLflow server:
```
docker-compose -f docker-compose.yaml -f docker-compose-local.yaml  up -d
```