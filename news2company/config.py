logging_config = {
    # see https://docs.python.org/3/library/logging.html#logging-levels for more information
    'level': 'INFO',  # logging level
    'format': '%(asctime)s [%(levelname)s]\n%(message)s\n',  # logging format
    'datefmt': '%Y-%m-%d %H:%M:%S',  # date format
    'file': 'news2company.log',  # file to write log
    'stdout': True,  # whether output to stdout
}

transfer_config = {
    'bucket': 'com.miotech.data.prd',  # bucket for storing result
    'model': {
        'upstream': 'Project/miotech/miotech-cn-nlp/news2company/model/',  # key to store model
        'local': '/tmp/ner_model',  # local dir for storing model on each executor
        'config_file': 'config_file',  # file name of model's config
        'map_file': 'maps.pkl',  # file name of model's mapping
        'ckpt_path': 'ckpt',  # path name of model's checkpoints
        'word2vec': 'word2vec'  # file name of word2vec lookup table
    },
    'ner_table': 'dm.cn_narrative_ner_result',
    'quoted_table': 'dm.cn_narrative_ner_quoted_match',
    # 'quoted_unmatch': 'dw.ner_quoted_unmatched_result_tmp',
    # 'rule_table': 'dw.ner_tyc_rule_matched_tmp'
}
