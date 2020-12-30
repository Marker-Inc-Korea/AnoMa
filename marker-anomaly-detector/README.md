## Run

Set up docker file:
```
docker build --tag mad_anoma .
```

To lauch a anomaly train:
```
docker-compose -f docker-compose-mad.yml -p MAD_ANOMA up -d
```

To lauch a anomaly deploy:
```
docker-compose -f docker-compose-mad-deploy.yml -p MAD_ANOMADEPLOY up -d
```

