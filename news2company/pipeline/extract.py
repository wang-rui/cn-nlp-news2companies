import re

import pyspark.sql.functions as F
import pyspark.sql.types as T

from ner.evaluate import evaluate_news


def news2entity(news, save_to):
    def clean_content(c):
        c = re.sub(r'<.*?>', '', c).strip()
        c = re.sub('。+', '。', c)
        if c == '':
            return None
        return c

    def split_content(c):
        seg_ptn = re.compile(u'((?!“).*?[！？。]+(?!”))|(.*?[！？。…—]+”)')
        junk_ptn = re.compile(u'(^\?+)|(\s+)')
        sentences = []
        for match in re.finditer(seg_ptn, c):
            start = match.start()
            end = match.end()
            s = c[start: end]
            cleaned = re.sub(junk_ptn, '', s)
            sentences.append(cleaned)
        return sentences

    def clean_entity(entity):
        black_list = {'公司', '集团', '中心', '某公司', '某集团', '某中心'}
        entity = re.sub(r'\W+', '', entity)
        if re.sub(r'[0-9a-zA-Z.]', '', entity) == '':
            return None
        elif len(entity) < 2:
            return None
        elif entity in black_list:
            return None
        else:
            return entity

    def ner(data):
        for news_id, entity in evaluate_news(data):
            clean = clean_entity(entity)
            if clean:
                yield news_id, clean
        # todo: remove comment below after implement downloading model and creating model respectively
        # todo: move clean_entity to match step
        # mdl_path = g_model_path.value
        # l_mdl_path = mdl_path['local']
        # conf_file = mdl_path['config_file']
        # mp_file = mdl_path['map_file']
        # w2c = mdl_path['word2vec']
        # ckpt = mdl_path['ckpt_path']
        # l_conf_file = os.path.join(l_mdl_path, conf_file)
        # l_mp_file = os.path.join(l_mdl_path, mp_file)
        # l_w2c = os.path.join(l_mdl_path, w2c)
        # l_ckpt = os.path.join(l_mdl_path, ckpt)
        # if not os.path.exists(l_mdl_path):
        #     os.makedirs(l_mdl_path)
        #     u_mdl_path = mdl_path['upstream'].rstrip('/') + '/{0}'
        #     tm = TransferManager(g_bucket.value)
        #     tm.download_file(u_mdl_path.format(conf_file), l_conf_file)
        #     tm.download_file(u_mdl_path.format(mp_file), l_mp_file)
        #     tm.download_file(u_mdl_path.format(w2c), l_w2c)
        #     tm.download_dir_file(u_mdl_path.format(ckpt), l_ckpt)
        # tf_config = tf.ConfigProto()
        # tf_config.gpu_options.allow_growth = True
        # conf = load_config(l_conf_file)
        # with open(l_mp_file, 'rb') as f:
        #     char2id, id2char, tag2id, id2tag = pickle.load(f)
        # entities = defaultdict(set)
        # with tf.Session(config=tf_config) as sess:
        #     model = create_model(sess, Model, l_ckpt, load_word2vec, conf, id2char)
        #     for news_id, sentence in data:
        #         result = model.evaluate_line(sess, input_from_line(sentence, char2id), id2tag)
        #         for entity in result['entities']:
        #             e_type = entity['type']
        #             if e_type != 'ORG':
        #                 continue
        #             start = entity['start']
        #             end = entity['end']
        #             org = sentence[start: end]
        #             clean = clean_entity(org)
        #             if clean:
        #                 entities[news_id].add(clean)
        # for news_id, es in entities.items():
        #     for e in es:
        #         yield news_id, e
    #
    # g_model_path = spark.sparkContext.broadcast(model_path)
    # g_bucket = spark.sparkContext.broadcast(bucket)

    clean_content_udf = F.udf(lambda x: clean_content(x), T.StringType())
    split_content_udf = F.udf(lambda x: split_content(x), T.ArrayType(T.StringType()))
    clean_entity_udf = F.udf(lambda e: clean_entity(e), T.StringType())

    news.cache()

    id_time = news.select('news_id', 'publish_time')

    company = news.withColumn('contents',
                              F.concat_ws('。', F.col('news_title'), F.col('news_summary'), F.col('news_body'))) \
        .select('news_id', clean_content_udf('contents').alias('content')) \
        .filter(F.col('content').isNotNull()) \
        .withColumn('sentences', split_content_udf(F.col('content'))) \
        .select('news_id', F.explode('sentences').alias('sentence')) \
        .rdd \
        .mapPartitions(ner) \
        .toDF(['news_id', 'company']) \
        .select('news_id', clean_entity_udf('company').alias('company'))

    news_company = company.join(id_time, 'news_id') \
        .withColumn('process_time', F.current_timestamp())
    news_company.write.saveAsTable(save_to, mode='append')
    # todo: deal with publish time
