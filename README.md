# AnoMa

## Run

**To set AnoMa environment:**  
env_tenplate을 통해 .env를 생성 후 하위 폴더들에 .env  파일을 복사함.

```
cp env_template .env
vi .env # make chanege Anoma environment
sh copy_env.sh
```

### 1. grafana_db
Grafana dashboard와 InfluxDB Docker container를 생성함.

**To set Grafana influxDB:**  

```
mkdir anoam_grafana
mkdir anoma_influx_db
```

**To launch grafana influx db:**
```
docker-compose -f docker-compose.yml up -d
```

### 2. marker-anomaly-detector


**Set up docker file:**
```
docker build --tag mad_anoma .
```

**To launch anomaly train:**  
train interval에 따라 모델 학습이 이뤄지는 Docker container 생성
```
docker-compose -f docker-compose-mad.yml -p MAD_ANOMA up -d
```

**To launch anomaly deploy:**  
실시간으로 DB의 최신 값을 불러와 Anomaly 예측이 이뤄지는 Docker container 생성
```
docker-compose -f docker-compose-mad-deploy.yml -p MAD_ANOMADEPLOY up -d
```

**Docker 환경변수**  

|변수|내용|Default|
|:------|:------|------:|
|FLT_INFL_URL|InfluxDB URL 주소|
|FLT_RETRAINING_INTERVAL_MINUTES|스케쥴러 내장 함수를 통한 재학습 주기, 단위 : 분|default = 120 |
|FLT_ROLLING_TRAINING_WINDOW_SIZE|Train 데이터 구간의 크기, <br/>현재로 부터 몇일 뒤의 데이터를 불러올 것인지 결정|default = 3d|
|FLT_DATABASE_NAME|사용하는 InfluxDB의 DB 이름||
|FLT_DATABASE_PORT|InfluxDB의 포트, .env 환경변수 설정 파일로 부터 <br/>INFLUX_DB_PORT 변수 값을 가져옴||
|FLT_DATABASE_USERNAME|InfluxDB의 ID, .env 환경변수 설정 파일로 부터 <br/>INFLUX_DB_USER_NAME 변수 값을 가져옴||
|FLT_DATABASE_PASSWORD|InfluxDB의 Password, .env 환경변수 설정 파일로 부터 <br/>INFLUX_DB_PASSWORD 변수 값을 가져옴||
|FLT_ALL_TABLES|전체 테이블을 불러오는 경우(True)|default = False|
|FLT_TABLE_LIST|특정 테이블 사용시 테이블 이름들을 세미콜론(;) 으로 구분하여 작성 <br/> ex. some_metric;ano_metric||
|FLT_ALL_COLUMNS|테이블의 전체 컬럼들을 불러오는 경우(True)|default = False|
|FLT_COLUMN_LIST|테이블의 특정 컬럼을 사용시 컬럼이름들을 dict형태로 작성 <br/> ex. {'some_metric':['val1', 'val2'],'ano_metric':['val11', 'val22']} ||
|MODEL_DIR|marker-anomaly-detector 폴더 하위에 모델이 저장될 폴더 이름 <br/> 경로상 폴더가 없을시 자동생성||






