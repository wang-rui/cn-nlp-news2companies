import re
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.utils import AnalysisException


def quoted_company_match(spark, ner_table, time_bg, time_end, quoted_match_table, logger):
    def include_match(x):
        company = global_descending_quoted_company_list.value
        for name, party_id in company:
            if name in x:
                return party_id
        return None
    include_match_udf = F.udf(lambda x: include_match(x), T.LongType())

    logger.info("Reading in news...")
    news = spark.read.table(ner_table) \
        .filter("publish_time > TIMESTAMP '{0}' AND publish_time < TIMESTAMP '{1}'".format(time_bg, time_end)) \
        .select('news_id', 'company', 'publish_time')
    news.cache()
    logger.info("Read in news.")

    logger.info("Reading in company data...")
    quoted_company = spark.read.table('dm.cn_security_data') \
        .filter(F.col('ticker_symbol').isNotNull()) \
        .select(F.col('sec_full_name').alias('company_full_name'),
                F.col('sec_short_name').alias('company_short_name'),
                'party_id')
    logger.info("Read in company data.")

    ner_company_name = news.select('company').dropDuplicates()

    quoted_company_list = quoted_company.rdd \
        .flatMap(lambda x: [(x[0], x[2]), (x[1], x[2])]) \
        .collect()
    descending_quoted_company_list = sorted(quoted_company_list, key=lambda x: len(x[0]), reverse=True)
    global_descending_quoted_company_list = spark.sparkContext.broadcast(descending_quoted_company_list)

    ner_company_name = ner_company_name.withColumn('party_id', include_match_udf('company').cast('bigint')) \
        .filter(F.col('party_id').isNotNull())

    # include_unmatched_news = news.join(ner_company_name, news['company'] == ner_company_name['company'], 'left_anti')

    include_quoted_company_matched = news.join(ner_company_name, news['company'] == ner_company_name['company']) \
        .select(news['news_id'], news['company'], news['publish_time'], ner_company_name['party_id']) \


    try:
        origin = spark.read.table(quoted_match_table)
    except AnalysisException as e:
        # no save_to table
        logger.warning(e)
        logger.info("Saving quoted company match result...")
        include_quoted_company_matched.write.saveAsTable(quoted_match_table, mode='overwrite')
        logger.info("Created table {0} and saved quoted company match result to it.".format(quoted_match_table))
        spark.catalog.clearCache()
        return

    tmp_table = quoted_match_table + '_tmp'
    logger.info("Start de-duplicating quoted company match result...")
    de_dup = origin.select('news_id', 'company', 'party_id', 'publish_time') \
        .join(include_quoted_company_matched, 'news_id', 'left_anti')
    quoted_company_matched = de_dup.unionByName(include_quoted_company_matched)
    quoted_company_matched.write.saveAsTable(tmp_table, mode='overwrite')
    logger.info("De-duplicated quoted company match result")

    logger.info("Saving quoted company match result...")
    spark.read.table(tmp_table).write.saveAsTable(quoted_match_table, mode='overwrite')
    logger.info("Saved quoted company match result into {0}".format(quoted_match_table))

