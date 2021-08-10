import pandas as pd
import numpy as np
import warnings
warnings.simplefilter('ignore')
from tqdm import tqdm
from scipy import sparse as sp
import copy
from math import log
import pickle
from sklearn.metrics.pairwise import euclidean_distances
import ast
from config import *


def correct_line(line):
    
    line = line.replace('\n','').replace('\t','').replace('#','').replace('-','').replace('_','')
    line = line.lower()
    
    line = line.replace('heart','heartt').replace('art ',' ').replace('heartt','heart')
    
    line = line.replace('2d',' 2d ').replace('3d',' 3d ').replace('4d',' 4d ')
    line = line.replace('digitalpainting','digital')
    
    line = ' '+line+' '
    line = line.replace(' ai ',' AI ').replace('artificialintelligence','AI').replace('artificial intelligence','AI')
    line = line.replace(' vr ',' VR ').replace('virtualreality','VR').replace('virtual reality','VR')
    line = line.replace(' ml ',' ML ').replace('machinelearning','ML').replace('machine learning','ML')
    line = line.replace(' dl ',' DL ').replace('deeplearning','ML').replace('deep learning','DL')
    line = line.replace(' ar ',' AR ').replace('augmentedreality','AR').replace('augmented reality','AR')
    line = line.replace(' nn ',' NN ').replace('neuralnetwork','NN').replace('neural network','NN')
    line = line.replace(' gan ',' GAN ').replace('generative','GAN')
    
    return(line)


def get_similar_tags(tags, ind):

    #usd = tags.loc[ind]['usd']
    #tags[(tags['usd']>=usd/usd_range)&(tags['usd']<=usd*usd_range)]
    
    sum_tags = tags[list(tag_name.keys())]+tags.loc[ind][list(tag_name.keys())]
    overlaps = np.sum(sum_tags==2, axis=1)
    disparities = np.sum(sum_tags==1, axis=1)
    overlaps = pd.concat([overlaps, disparities], axis=1)
    overlaps.columns = ['overlaps','disparities']
    overlaps.sort_values(['overlaps','disparities'], ascending=[False,True], inplace=True)
    overlaps_inds = list(overlaps[overlaps['overlaps']>0].index)
    overlaps_inds = [x for x in overlaps_inds if x!=ind]
    
    return (str(list(tags.loc[overlaps_inds[:top_k]]['tokenId'])))


def get_bert_embeddings(tokens, bert, col):
    
    line_embeddings = []

    lines_list = tokens[~tokens[col].isnull()][col].values.tolist()
    lines_list = [' '.join(line.split()[:max_words]) for line in lines_list]
    lines_list = [' '.join([x for x in line.split() if not('http' in x)]) for line in lines_list]
    lines_list = [(line.replace('(','').replace(')','').replace('[','').replace(']','').replace('{','').replace('}','').replace('*','')
                   .replace('|','').replace('\\','').replace('/','').replace('_','').replace('+','').replace('-','').replace('=','')) 
                  for line in lines_list]
    lines_list = [' '.join(x for x in line.split() if (not(any(char.isdigit() for char in x)))&
                           ((x=='a')|((len(x)==2)&(not(x[-1] in ['.',',',';'])))|(len(x)>2))) for line in lines_list]
    lines_list = [line.lower() for line in lines_list]
    lines_list = [' '.join([x for x in line.split() if len(x)<=max_word_len]) for line in lines_list]

    for i in tqdm(range(len(lines_list)//bert_batch_size+1)):
        _, _, _, _, _, sent_mean_embs, _ = bert(lines_list[i*bert_batch_size:(i+1)*bert_batch_size])
        line_embeddings += [sent_mean_embs]
    
    line_embeddings = np.concatenate(line_embeddings)   
    
    return(line_embeddings)


def get_embeddings_recs(tokens, embeddings, col):
    
    token_recs = copy.deepcopy(tokens[~tokens[col].isnull()][['tokenId']])
    token_recs.index = range(token_recs.shape[0])

    tokenIds = tokens[~tokens[col].isnull()]['tokenId']

    dist = euclidean_distances(embeddings)

    for token in range(dist.shape[0]):
        token_recs.loc[token, 'top_tokens'] = str(tokenIds[np.argsort(dist[token,:]).tolist()[1:top_k+1]].tolist())
        
    return(token_recs)


def bm25_weighting(data_train):
    nonzero_sku = np.sum(data_train>0, axis=0)

    idf = {}
    for sku in list(nonzero_sku.index):
        idf[sku] = log((data_train.shape[0]-nonzero_sku[sku]+0.5)/(nonzero_sku[sku]+0.5))
    
    d = np.sum(data_train, axis=1)    
    avgdl = np.mean(d)
    tf = data_train.div(d, axis=0)

    data_train = tf.multiply(pd.Series(idf), axis=1)*(k1+1)/tf.add(k1*(1-b+b*(d/avgdl)), axis=0)
    return(data_train)


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

        
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
    
def get_interactions_tables(sales, bids):
    
    interactions = pd.concat([sales[['tokenId','buyer']].drop_duplicates().rename(columns={'buyer':'userid'}),
                              bids[['tokenId', 'bidder']].drop_duplicates().rename(columns={'bidder':'userid'})])
    interactions = interactions.drop_duplicates()

    stats = interactions.groupby('userid').size().reset_index()
    stats.rename(columns={0: 'count'}, inplace=True)
    interactions = interactions[interactions['userid'].isin(list(stats[stats['count']>=min_user_items]['userid'].values))]

    stats = interactions.groupby('tokenId').size().reset_index()
    stats.rename(columns={0: 'count'}, inplace=True)
    rare_products = list(stats[stats['count']<rare_product_count]['tokenId'].values)
    interactions = interactions[~interactions['tokenId'].isin(rare_products)]

    interactions['flag'] = 1
    data_full = pd.crosstab(interactions['userid'], interactions['tokenId'], interactions['flag'], aggfunc='sum')

    random.seed(1)

    test_indx = random.sample(list(interactions.index), round(len(interactions.index)*test_ratio))
    interactions.loc[test_indx,'flag'] = 0
    data_train = pd.crosstab(interactions[interactions['flag']==1]['userid'], interactions[interactions['flag']==1]['tokenId'],
                             interactions[interactions['flag']==1]['flag'], aggfunc='sum')

    data_test_flag = copy.deepcopy(data_full)
    data_test_flag[data_train>0] = -1

    data_full = bm25_weighting(data_full)
    data_full = data_full.fillna(0)
    data_train = bm25_weighting(data_train)
    data_train = data_train.fillna(0)

    used_tokenIds = list(data_full.columns)
    used_users = list(data_full.index)

    data_train = data_train.values.T
    data_full = data_full.values.T
    data_test_flag = data_test_flag.values.T

    data_train = sp.csr_matrix(data_train)
    
    return(data_full, data_train, data_test_flag, used_tokenIds, used_users)


def get_implicit_recs(used_users, used_tokenIds, preds_new, item_embeddings):
    
    user_recs_implicit = pd.DataFrame(used_users, columns=['userId'])
    token_recs_implicit  = pd.DataFrame(used_tokenIds, columns=['tokenId'])

    used_users = np.array(used_users)
    used_tokenIds = np.array(used_tokenIds)    
    
    for user in range(preds_new.shape[1]):
        user_recs_implicit.loc[user,'top_tokens'] = str(used_tokenIds[list(np.argsort(-preds_new[:,user])[:top_k])].tolist())    
    
    dist = euclidean_distances(item_embeddings)
    for token in range(dist.shape[0]):
        token_recs_implicit.loc[token,'top_tokens'] = str(used_tokenIds[np.argsort(dist[token,:]).tolist()[1:top_k+1]].tolist())    
        
    return(user_recs_implicit, token_recs_implicit)


def get_user_preds(interactions, token_recs_col, col):
    user_recs_col = interactions[['userId']].drop_duplicates()
    user_recs_col.index = range(user_recs_col.shape[0]) 
    
    for i, user in tqdm(enumerate(user_recs_col['userId'].tolist())):
        top_ks = []
        old_tokens = interactions[(interactions['userId']==user)]['tokenId'].tolist()
    
        for token in old_tokens:
            try:
                top_ks += [ast.literal_eval(token_recs_col[token_recs_col['tokenId']==token][col].values[0].replace('nan,','-1,').replace(', nan',', -1'))]
            except:
                None            
        
        top_ks = [top for top in top_ks if len(top)!=0]
        top_ks_ordered = [top[i] for i in range(top_k) for top in top_ks]
        used = set()
        top_ks_ordered = [x for x in top_ks_ordered if x not in used and (used.add(x) or True)]
        top_ks_ordered = [x for x in top_ks_ordered if not(x in old_tokens)]
    
        user_recs_col.loc[i, 'top_tokens']  = str(top_ks_ordered[:top_k])
        
    return(user_recs_col)


def get_preds_compilation(col_id, recs_tags, recs_tags_2, recs_description, recs_implicit):
    
    recs = pd.concat([recs_tags[[col_id]], recs_tags_2[[col_id]], recs_description[[col_id]], recs_implicit[[col_id]]]).drop_duplicates()

    recs = pd.merge(recs, recs_tags, on=col_id, how='left')
    recs = pd.merge(recs, recs_tags_2, on=col_id, how='left')
    recs = pd.merge(recs, recs_description, on=col_id, how='left')
    recs = pd.merge(recs, recs_implicit, on=col_id, how='left')
    
    recs = recs.where(~recs.isnull(), other='[]')
    
    for i in recs.index:
        (tags_tokens, tags_2_tokens, description_tokens, collaborative_tokens) = recs.loc[i][['tags_tokens', 'tags_2_tokens',
                                                                                                    'description_tokens', 'collaborative_tokens']]
    
        tags_tokens = ast.literal_eval(tags_tokens)
        tags_2_tokens = ast.literal_eval(tags_2_tokens.replace('nan,','-1,').replace(', nan',', -1'))
        description_tokens = ast.literal_eval(description_tokens.replace('nan,','-1,').replace(', nan',', -1'))
        #description_tokens = [x for x in description_tokens if not(x==-1)]
        collaborative_tokens = ast.literal_eval(collaborative_tokens)
    
        top_ks = [top for top in [tags_tokens, tags_2_tokens, description_tokens, collaborative_tokens] if len(top)!=0]
        top_ks = [top+(top_k-len(top))*[-1] for top in top_ks]
        top_ks_ordered = [top[i] for i in range(top_k) for top in top_ks]
        used = set()
        top_ks_ordered = [x for x in top_ks_ordered if x not in used and (used.add(x) or True)]
        top_ks_ordered = [x for x in top_ks_ordered if x!=-1]
    
        recs.loc[i, 'compilation_tokens'] = str(top_ks_ordered[:top_k])   
    
    return(recs)

