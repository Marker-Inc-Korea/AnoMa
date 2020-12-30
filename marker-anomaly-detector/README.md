## Run

Set up docker file:
```
docker build --tag pad_anoma .
```

To lauch a anomaly deploy:
```
docker-compose -f docker-compose-pad.yml -p ANOMA_PAD up -d
```

To lauch a anomaly testing:
```
docker-compose -f docker-compose-pad.yml -p ANOMA_TESTING up -d
```