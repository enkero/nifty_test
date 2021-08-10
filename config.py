###   Length of recommendation lists   ####

top_k = 30


###   Filenames   ####

filename_bids = 'data/bids.csv'
filename_sales = 'data/sales.csv'
filename_tokens = 'data/tokens.csv'

filename_recs_tags = 'recs/token_recs_tags.csv'
filename_recs_description = 'recs/token_recs_description.csv'
filename_recs_tags_2 = 'recs/token_recs_tags_2.csv'
filename_recs_implicit = 'recs/token_recs_implicit.csv'
filename_user_recs_implicit = 'recs/user_recs_implicit.csv'
filename_user_recs_tags = 'recs/user_recs_tags.csv'
filename_user_recs_tags_2 = 'recs/user_recs_tags_2.csv'
filename_user_recs_description = 'recs/user_recs_description.csv'
filename_recs = 'recs/token_recs.csv'
filename_user_recs = 'recs/user_recs.csv'

filename_item_embeddings = 'models/item_embeddings.npy'
filename_user_embeddings = 'models/user_embeddings.npy'
#filename_used_tokenIds = 'models/used_tokenIds'
#filename_used_users = 'models/used_users'
filename_model_implicit = 'models/model_implicit'

filename_item_embeddings_new = 'models/item_embeddings_' + str(datetime.date.today()) +'.npy'
filename_user_embeddings_new = 'models/user_embeddings_' + str(datetime.date.today()) +'.npy'
#filename_used_tokenIds_new = 'models/used_tokenIds_' + str(datetime.date.today())
#filename_used_users_new = 'models/used_users_' + str(datetime.date.today())
filename_model_implicit_new = 'models/model_implicit_' + str(datetime.date.today())


###   BERT configs   ####

bert_path = 'models/conversational_cased_L-12_H-768_A-12_pt'
bert_batch_size = 10
max_words = 300
max_word_len = 20


###   Collaborative filtering configs   ####

min_user_items = 2
rare_product_count = 2

k1 = 1.5  
b = 0.75  

test_ratio = 0.3
dim_embeddings = 30
reg = 0.002
iters = 5
train_model = False
check_accuracy = False


###   Tag categories   ####

tag_name = {}
tag_name['3D'] = ['3d']

tag_name['gif_video'] = ['gif','animated','video','motion','animatedgif','movement','animation']

tag_name['painting'] = ['painting','drawing','sketch','illustration']

tag_name['ML'] = ['AI','ML','DL','NN','GAN','ganbreeder','artbreeder','deepstyle','deepdreamgenerator','deep','neural','algorithmic',
                  'deepdreamgenerator','biggan','geneticalgorithm']

tag_name['AR_VR'] = ['AR','VR']

tag_name['cryptocurrency'] = ['bitcoin','ethereum','blockchain','eth','nfts','nft','btc','token','tokens','cryptocurrency','money','currency',
                              'nonfungible','fiat','dollar','coin','hodl','oldmoneycorrupts']

tag_name['black_white'] = ['monochrome','blackandwhite','black','white']

tag_name['color'] = ['colors','color','colour','colorful','blue','red','gold','green','neon','pink','yellow','purple','orange','iridescent']

tag_name['psychedelic'] = ['surreal','psychedelic','fractal','spiritual','surrealism','trippy','soul','lsd']

tag_name['space'] = ['space','moon','universe','earth','stars','cosmic','sun','planet','astronaut']

tag_name['abstract'] = ['abstract','abstraction','abstracts']

tag_name['future'] = ['future','futuristic','futurism','scifi','dystopia','cyberpunk']

#tag_name['design'] = ['design','architecture']

tag_name['sculpture'] = ['sculpture','statue']

tag_name['glitch'] = ['glitch','noise']

tag_name['human'] = ['human','woman','girl','female','people','man','women','face','portrait','selfportrait']


#usd_range = 3







