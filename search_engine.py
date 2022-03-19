from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle
import pandas as pd
from predict import *

model = SentenceTransformer('bert-base-nli-mean-tokens')

with open('/home/moulisai_2296/search_relevence/search_relevence/search_engine_utils/sentence_embeddings.pkl', 'rb') as f:
    sentence_embeddings = pickle.load(f)

df_similar_data = pd.read_pickle('/home/moulisai_2296/search_relevence/search_relevence/search_engine_utils/similarity.pkl')

def similar_titles(search_term):
    search = [search_term]
    search_term_embedding = model.encode(search[0])
    cos_sim = cosine_similarity([search_term_embedding[0]],sentence_embeddings[0:]) #cosine values between search term and titles
    df_similar_data = pd.read_pickle('/home/moulisai_2296/search_relevence/search_relevence/search_engine_utils/similarity.pkl')
    df_similar_data['cosine'] = cos_sim[0]
    df_similar_data = df_similar_data.sort_values(by=['cosine'], ascending=False)
    df_top50 = df_similar_data[0:100]
    df_top50.drop(['merge'],axis=1,inplace=True)
    df_top50.drop(['product_description'],axis=1,inplace=True)
    df_top50.drop(['product_brand'],axis=1,inplace=True)
    s = search[0] * 100
    df_top50['search_term'] = s
    return df_top50

def search_sim(search_term):
    df_top50 = similar_titles(search_term)
    print("check1")
    title = df_top50['product_title'].values
    search = df_top50['search_term'].values
    example = []
    for i in range(0, len(title)):
        example.append([title[i],search[i]])
    print("check2")
    model_test = predict_test(example)[1]
    print("check 3")
    print(model_test)
    model_test = model_test.sort_values(by=['avg'], ascending=False)
    titles = model_test['title'].values
    rel = model_test['avg'].values
    rel = rel[0:6]
    titles = titles[0:6]
    title_dict = dict()
    title_dict[titles[0]] = rel[0]
    title_dict[titles[1]] = rel[1]
    title_dict[titles[2]] = rel[2]
    title_dict[titles[3]] = rel[3]
    title_dict[titles[4]] = rel[4]
    
    return title_dict
