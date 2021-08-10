import pandas as pd
import numpy as np
import warnings
warnings.simplefilter('ignore')
from tqdm import tqdm
import copy
from config import *
from functions import *


###   Read data   ####

bids = pd.read_csv(filename_bids)
sales = pd.read_csv(filename_sales)
tokens = pd.read_csv(filename_tokens)

bids['timestamp'] = pd.to_datetime(bids['timestamp'])
bids.sort_values('timestamp', inplace=True)

sales['timestamp'] = pd.to_datetime(sales['timestamp'])
sales.sort_values('timestamp', inplace=True)

tokens['timestamp'] = pd.to_datetime(tokens['timestamp'])
tokens.sort_values('timestamp', inplace=True)


###   Remove doubles   ####

stats = tokens[['tokenId']].groupby('tokenId').size().reset_index()
stats.rename(columns={0: 'count'}, inplace=True)
tokens.drop(tokens[tokens['tokenId'].isin(stats[stats['count']>=2]['tokenId'])][['tokenId']].drop_duplicates(keep='last').index, inplace=True)


###   Compute interaction matrices for train and validation   ####

(data_full, data_train, data_test_flag, used_tokenIds, used_users) = get_interactions_tables(sales, bids)


###   Train model or load previously calculated embeddings   ####

if train_model:
    
    import implicit

    model = implicit.als.AlternatingLeastSquares(factors=dim_embeddings, regularization=reg, iterations=iters)
    model.fit(data_train)
    
    item_embeddings = model._item_factor(list(range(data_train.get_shape()[0])), [], recalculate_item=False)
    user_embeddings = model._user_factor(list(range(data_train.get_shape()[1])), []) 
    
    np.save(filename_item_embeddings_new, item_embeddings)
    np.save(filename_user_embeddings_new, user_embeddings)
    #save_obj(used_tokenIds, filename_used_tokenIds_new)
    #save_obj(used_users, filename_used_users_new)
    save_obj(model, filename_model_implicit_new)
    
else:
    
    #model = load_obj(filename_model_implicit)    
    #used_tokenIds = load_obj(filename_used_tokenIds)
    #used_users = load_obj(filename_used_users)
    item_embeddings = np.load(filename_item_embeddings)
    user_embeddings = np.load(filename_user_embeddings)
    

###   Get prediction matrix   ####

preds = np.matmul(item_embeddings, user_embeddings.T)
preds_new = copy.deepcopy(preds)
preds_new[data_test_flag==-1] = np.min(preds)-1


###   Check accuracy on validaion (mainly for new models)   ####

if check_accuracy:
    
    top_k_accuracy = []
    top_k_accuracy_new = []

    for user in tqdm(range(preds.shape[1])):
        k = np.min([top_k, np.sum(data_full[:,user]>0)])
        if k>0:
            top_k_accuracy += [np.sum(data_full[list(np.argsort(-preds[:,user])[:k]),user]>0)/k]       
        k = np.min([top_k, np.sum(data_test_flag[:,user]>0)])
        if k>0:
            top_k_accuracy_new += [np.sum(data_full[list(np.argsort(-preds_new[:,user])[:k]),user]>0)/k]
    
    print(np.nanmean(top_k_accuracy), np.nanmean(top_k_accuracy_new))  
    

###   Get collaborative recommendations   ####
    
(user_recs_implicit, token_recs_implicit) = get_implicit_recs(used_users, used_tokenIds, preds_new, item_embeddings)    
    
user_recs_implicit.to_csv(filename_user_recs_implicit, index=False)
token_recs_implicit.to_csv(filename_recs_implicit, index=False)    
    
    









