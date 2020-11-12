## Run

To set a prometheus environment:
```
cp prometheus_template.yml prometheus.yml
vi prometheus.yml # make chanege prometheus environment
```
To set a target_push & target_push json file:
```
cp target_push_template tartet_push.json
vi target_push.json # make chanege target metric info
```
To set a target_push & target_pad json file:
```
cp target_pad_template tartet_pad.json
vi target_push.json # make chanege target metric info
```

To lauch prometheus
```
docker-compose -f docker-compose.yml up -d
```