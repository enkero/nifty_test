import pandas as pd
import numpy as np
import warnings
warnings.simplefilter('ignore')
from deeppavlov.core.common.file import read_json
from deeppavlov import build_model, configs
from config import *
from functions import *


###   Read data   ####

tokens = pd.read_csv(filename_tokens)

tokens['timestamp'] = pd.to_datetime(tokens['timestamp'])
tokens.sort_values('timestamp', inplace=True)


###   Remove doubles   ####

stats = tokens[['tokenId']].groupby('tokenId').size().reset_index()
stats.rename(columns={0: 'count'}, inplace=True)
tokens.drop(tokens[tokens['tokenId'].isin(stats[stats['count']>=2]['tokenId'])][['tokenId']].drop_duplicates(keep='last').index, inplace=True)


###   Initialize pretrained BERT model   ####

bert_config = read_json(configs.embedder.bert_embedder)
bert_config['metadata']['variables']['BERT_PATH'] = bert_path

bert = build_model(bert_config)


###   Apply BERT model to descriptions and get recommendations   ####

descriptions_embeddings = get_bert_embeddings(tokens, bert, 'description')
token_recs_description = get_embeddings_recs(tokens, descriptions_embeddings, 'description') 
token_recs_description.to_csv(filename_recs_description, index=False)


###   Apply BERT model to tags and get recommendations   ####

tags_embeddings = get_bert_embeddings(tokens, bert, 'tags')
token_recs_tags_2 = get_embeddings_recs(tokens, tags_embeddings, 'tags') 
token_recs_tags_2.to_csv(filename_recs_tags_2, index=False)



