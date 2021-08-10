import pandas as pd
import numpy as np
import warnings
warnings.simplefilter('ignore')
from config import *
from functions import *


###   Read data   ####

bids = pd.read_csv(filename_bids)
sales = pd.read_csv(filename_sales)

token_recs_tags = pd.read_csv(filename_recs_tags)
token_recs_tags_2 = pd.read_csv(filename_recs_tags_2)
token_recs_description = pd.read_csv(filename_recs_description)
token_recs_implicit = pd.read_csv(filename_recs_implicit)
user_recs_implicit = pd.read_csv(filename_user_recs_implicit)

token_recs_tags.columns = ['tokenId', 'tags_tokens']
token_recs_tags_2.columns = ['tokenId', 'tags_2_tokens']
token_recs_description.columns = ['tokenId', 'description_tokens']
token_recs_implicit.columns = ['tokenId', 'collaborative_tokens']
user_recs_implicit.columns = ['userId', 'collaborative_tokens']


###   Make user recommendations   ####

interactions = pd.concat([sales[['tokenId','buyer']].drop_duplicates().rename(columns={'buyer':'userId'}),
                          bids[['tokenId', 'bidder']].drop_duplicates().rename(columns={'bidder':'userId'})])
interactions = interactions.drop_duplicates()

user_recs_tags = get_user_preds(interactions, token_recs_tags, 'tags_tokens')
user_recs_tags.to_csv(filename_user_recs_tags, index=False)

user_recs_tags_2 = get_user_preds(interactions, token_recs_tags_2, 'tags_2_tokens')
user_recs_tags_2.to_csv(filename_user_recs_tags_2, index=False)

user_recs_description = get_user_preds(interactions, token_recs_description, 'description_tokens')
user_recs_description.to_csv(filename_user_recs_description, index=False)


###   Combine all recommenddations and make compilations   ####

token_recs = get_preds_compilation('tokenId', token_recs_tags, token_recs_tags_2, token_recs_description, token_recs_implicit)
token_recs.to_csv(filename_recs, index=False)

user_recs_tags.columns = ['userId', 'tags_tokens']
user_recs_tags_2.columns = ['userId', 'tags_2_tokens']
user_recs_description.columns = ['userId', 'description_tokens']

user_recs = get_preds_compilation('userId', user_recs_tags, user_recs_tags_2, user_recs_description, user_recs_implicit)
user_recs.to_csv(filename_user_recs, index=False)











