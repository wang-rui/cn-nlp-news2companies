def query_news(spark, time_bg, time_end):
    content = spark.read.table('dw.tl_vnews_content_nondupbd') \
        .select('news_id', 'news_title', 'update_time') \
        .withColumnRenamed('update_time', 'publish_time')
    summary = spark.read.table('dw.tl_vnews_summary_v1') \
        .select('news_id', 'news_summary')
    body = spark.read.table('dw.tl_vnews_body_v1') \
        .select('news_id', 'news_body')

    news = content.join(summary, 'news_id') \
        .join(body, 'news_id') \
        .filter("publish_time > TIMESTAMP '{0}' AND publish_time < TIMESTAMP '{1}'"
                .format(time_bg, time_end))
    return news
