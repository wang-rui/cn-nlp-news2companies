# Miotech-CN-NLP

Init

## News2Company

Map news within a certain period to relevant companies.

### Environment

- spark 2.3
- python 3.4

#### python packages

- boto3
- tensorflow
- jieba
- retrying

### Setup

Under path miotech-cn-nlp/news2company:

```bash
$ zip -r dependency.zip pipeline
$ python3 client.py # show useage
usage: client.py [-h] [-s STARTDATE] [-e ENDDATE] [-t TABLE]

Map news within a certain period to relevant companies.

optional arguments:
  -h, --help            show this help message and exit
  -s STARTDATE, --startdate STARTDATE
                        Start update-date of news, format: year-month-day
  -e ENDDATE, --enddate ENDDATE
                        End update-date of news, format: year-month-day
  -t TABLE, --table TABLE
                        Table to save mapping result, format: database.table
$ spark-submit --py-files dependency.zip client.py [-s|--startdate] STARTDATE [-e|--enddate] ENDDATE [-t|--table] TABLE
```

*NOTE: Please check that the environment variable `PYSPAKR_DRIVER_PYTHON` set in `path/to/spark/conf/spark-env.sh` is commented, or the driver will not accept any argument when start up. Otherwise the argument passing to `client.py` might not be recognized by the script but by the driver.*

### Settings

You can change the settings in `config.py`