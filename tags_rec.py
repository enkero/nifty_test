import pandas as pd
import numpy as np
import warnings
warnings.simplefilter('ignore')
from tqdm import tqdm
import copy
from config import *
from functions import *


###   Read data   ####

#sales = pd.read_csv(filename_sales)
tokens = pd.read_csv(filename_tokens)

#sales['timestamp'] = pd.to_datetime(sales['timestamp'])
#sales.sort_values('timestamp', inplace=True)

tokens['timestamp'] = pd.to_datetime(tokens['timestamp'])
tokens.sort_values('timestamp', inplace=True)


###   Remove doubles   ####

stats = tokens[['tokenId']].groupby('tokenId').size().reset_index()
stats.rename(columns={0: 'count'}, inplace=True)
tokens.drop(tokens[tokens['tokenId'].isin(stats[stats['count']>=2]['tokenId'])][['tokenId']].drop_duplicates(keep='last').index, inplace=True)


###   Join price   ####

#inds_last_sale = sales[~sales['usd'].isnull()][['tokenId']].drop_duplicates(keep='last').index
#tokens = pd.merge(tokens, sales.loc[inds_last_sale][['tokenId','usd']], on='tokenId', how='left')
#tags = copy.deepcopy(tokens[['tokenId','usd','tags']])


###   Normalize tags text lines   ####

tags = copy.deepcopy(tokens[['tokenId','tags']])
tags.loc[~tags['tags'].isnull(),'tags'] = tags[~tags['tags'].isnull()]['tags'].apply(correct_line)


###   Define tag category   ####

for tag in tag_name.keys():
    tags.loc[~tags['tags'].isnull(), tag] = [int(len(set(tag_name[tag])&set(line.split()))>=1) for line in tags[~tags['tags'].isnull()]['tags']]

    
tags.index = range(tags.shape[0])
tags.fillna(0, inplace=True)


###   Get recommendations with similar tags   ####

for ind in tqdm(tags.index):    
    tags.loc[ind,'similar_tags'] = get_similar_tags(tags, ind)
    

tags[['tokenId','similar_tags']].to_csv(filename_recs_tags, index=False)




