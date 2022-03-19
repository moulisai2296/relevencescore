from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')

import pickle
import pandas as pd

df_similar_data = pd.read_pickle('/home/moulisai_2296/search_relevence/search_relevence/search_engine_utils/similarity.pkl')
list_sent = df_similar_data['merge'].values
print("sentences collected")
sentence_embeddings = model.encode(list_sent)  #array with 768 columns

pickle.dump(sentence_embeddings, open("/home/moulisai_2296/search_relevence/search_relevence/search_engine_utils/sentence_embeddings.pkl", "wb"))
print("DONE")
