import datetime
import argparse
import logging

from config import transfer_config as tc, logging_config as lc


def get_yesterday_str():
    t = datetime.date.today()
    d1 = datetime.timedelta(days=1)
    return datetime.date.isoformat(t - d1)


def validate(d):
    try:
        datetime.datetime.strptime(d, '%Y-%m-%d')
        return d
    except ValueError:
        msg = "Not a valid date format: '{0}'.".format(d)
        raise argparse.ArgumentTypeError(msg)


if __name__ == '__main__':
    # arg parser
    parser = argparse.ArgumentParser(description="Map news within a certain period to relevant companies.")
    parser.add_argument('-s', '--startdate',
                        help="Start update-date of news, format: year-month-day",
                        default=get_yesterday_str(),
                        type=validate)
    parser.add_argument('-e', '--enddate',
                        help="End update-date of news, format: year-month-day",
                        default=datetime.date.isoformat(datetime.date.today()),
                        type=validate)
    parser.add_argument('--nertable',
                        help="Table to save ner result, format: database.table",
                        default=tc['ner_table'],
                        type=str)
    parser.add_argument('--quotedtable',
                        help="Table to save quoted company match result, format: database.table",
                        default=tc['quoted_table'],
                        type=str)
    args = parser.parse_args()

    from pyspark.sql import SparkSession
    from pyspark.conf import SparkConf
    from pipeline.extract import news2entity
    from pipeline.query import query_news
    from pipeline.transfer import TransferManager
    from pipeline.quoted_company_match import quoted_company_match

    time_begin = args.startdate + ' 00:00:00'
    time_end = args.enddate + ' 00:00:00'

    # logging config
    logging_handlers = [logging.FileHandler(lc['file'])]
    if lc['stdout']:
        logging_handlers.append(logging.StreamHandler())
    logging.basicConfig(level=lc['level'].upper(),
                        format=lc['format'],
                        datefmt=lc['datefmt'],
                        handlers=logging_handlers)
    logger = logging.getLogger()

    # file transfer
    trans_manager = TransferManager(tc['bucket'])

    # spark
    spark = SparkSession.builder \
        .appName('NEWS_ZH_NER_EXACTLY_MATCH') \
        .config(conf=SparkConf()) \
        .enableHiveSupport() \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    logging.info("Query news time range from {0} to {1}".format(time_begin, time_end))

    try:
        logger.info("Querying news...")
        news = query_news(spark, time_begin, time_end)
        logger.info("Queried news.")

        logger.info("Extracting companies from sentences...")
        news2entity(news=news,
                    save_to=args.nertable)
        logger.info("Extracted companies.")

        logger.info("Start quoted company matching...")
        quoted_company_match(spark=spark,
                             ner_table=args.nertable,
                             time_bg=time_begin,
                             time_end=time_end,
                             quoted_match_table=args.quotedtable,
                             logger=logger)
        logger.info("Completed quoted company matching.")

    except Exception as e:
        logger.error(e)
