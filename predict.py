import warnings
warnings.filterwarnings("ignore")
import plotly.graph_objs as go
import pandas as pd
from plotly.offline import iplot
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
from collections import Counter
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import plotly.graph_objs as go
from plotly.offline import iplot
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy.linalg import norm
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import math
from tqdm import tqdm
import nltk
nltk.download('stopwords')

df_titles_unique = pd.read_csv("/home/moulisai_2296/search_relevence/search_relevence/csv_files/titles_uids_unique (1).csv")
df_train_desc = pd.read_csv('/home/moulisai_2296/search_relevence/search_relevence/csv_files/product_descriptions.csv')
df_train_brand = pd.read_csv('/home/moulisai_2296/search_relevence/search_relevence/csv_files/attributes.csv')

bow_title = pd.read_pickle('/home/moulisai_2296/search_relevence/search_relevence/feature_fits/product_title_bow.pkl')
bow_desc = pd.read_pickle('/home/moulisai_2296/search_relevence/search_relevence/feature_fits/product_desc_bow.pkl')
bow_brand = pd.read_pickle('/home/moulisai_2296/search_relevence/search_relevence/feature_fits/product_brand_bow.pkl')
bow_search = pd.read_pickle('/home/moulisai_2296/search_relevence/search_relevence/feature_fits/search_term_bow.pkl')

#tfidf_title = pd.read_pickle('/content/drive/MyDrive/homedepot/feature_fits/product_title_tfidf.pkl')
#tfidf_desc = pd.read_pickle('/content/drive/MyDrive/homedepot/feature_fits/product_desc_tfidf.pkl')
#tfidf_brand = pd.read_pickle('/content/drive/MyDrive/homedepot/feature_fits/product_brand_tfidf.pkl')
#tfidf_search = pd.read_pickle('/content/drive/MyDrive/homedepot/feature_fits/search_term_tfidf.pkl')

tfidfw2v_title = pd.read_pickle('/home/moulisai_2296/search_relevence/search_relevence/feature_fits/product_title_tfidfw2vparam.pkl')
tfidfw2v_desc = pd.read_pickle('/home/moulisai_2296/search_relevence/search_relevence/feature_fits/product_desc_tfidfw2vparam.pkl')
tfidfw2v_brand = pd.read_pickle('/home/moulisai_2296/search_relevence/search_relevence/feature_fits/product_brand_tfidfw2vparam.pkl')
tfidfw2v_search = pd.read_pickle('/home/moulisai_2296/search_relevence/search_relevence/feature_fits/search_term_tfidfw2vparam.pkl')

RF = pd.read_pickle(('/home/moulisai_2296/search_relevence/search_relevence/model_files/RF_model.pkl'))
XGB = pd.read_pickle(('/home/moulisai_2296/search_relevence/search_relevence/model_files/DT_model.pkl'))
GBDT = pd.read_pickle(('/home/moulisai_2296/search_relevence/search_relevence/model_files/GBDT_model.pkl'))

linear_s1 = pd.read_pickle('/home/moulisai_2296/search_relevence/search_relevence/model_files/linear_reg_model_set1.pkl')
linear_s4 = pd.read_pickle('/home/moulisai_2296/search_relevence/search_relevence/model_files/linear_reg_model_set4.pkl')
linear_s5 = pd.read_pickle('/home/moulisai_2296/search_relevence/search_relevence/model_files/linear_reg_model_set5.pkl')

with open('/home/moulisai_2296/search_relevence/search_relevence/csv_files/glove_vectors', 'rb') as f:
    model = pickle.load(f)
    glove_words = set(model.keys())

def words(text): return re.findall(r'\w+', text.lower())
WORDS = Counter(words(open(r'/home/moulisai_2296/search_relevence/search_relevence/csv_files/corpus.txt',encoding='utf-8').read()))

#def test(test_example) :
#    test_df = pd.DataFrame(columns=['product_title', 'search_term'])
#    titles = []
#    print(test_example)
#    for i in test_example:
#        titles.append(i[0])
#    if len(test_example) == 1:
#        test_df.loc[len(test_df.index)] = test_example[0]
#    else:
#        test_df.loc[len(test_df.index)] = test_example[0]
#        t = []
#        for i in range(1, len(test_example)):
#            t = test_example[i]
#            test_df.loc[len(test_df.index)] = t
#    d = []
#
#    for title in titles:
#        for i in range(0, len(df_titles_unique.product_title)):
#            if title == df_titles_unique.product_title[i]:
#                index = i
#        d.append(list(df_titles_unique.iloc[index]))
#    uid = []
#    for i in d:
#        uid.append(i[0])
#    test_df['product_uid'] = uid
#
#    return test_df

def test(test_example) :
    dd = df_titles_unique.set_index('product_title').to_dict()['product_uid']
    test_df = pd.DataFrame(columns=['product_uid','product_title', 'search_term'])
    titles = []
    search = []
    uid = []
    for i in test_example:
        titles.append(i[0])
    for i in test_example:
        search.append(i[1])
    for i in titles:
        uid.append(dd[i])
    test_df['product_uid'] = uid
    test_df['product_title'] = titles
    test_df['search_term'] = search
    return test_df

def merge_test(test_df):
    #Merge product_descriptions with train data on product_uid
    df_train_with_desc = pd.merge(test_df, df_train_desc, on='product_uid', how='left')
    #Merge product_brand with train data on product_uid

    brand = df_train_brand[df_train_brand.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "product_brand"})
    df_final = pd.merge(df_train_with_desc, brand, on="product_uid", how="left")
    return df_final

def null_check(test_input):
    null_rows_test = test_input[test_input.isnull().any(1)]
    if null_rows_test.shape[0]==0:
        return test_input
    else:
        for i, row in null_rows_test.iterrows():
            t = row['product_title']
            t = t.split()
            null_rows_test['product_brand'].loc[i] = t[0]
        test_input['product_brand'].loc[null_rows_test.index] = null_rows_test['product_brand'].values
        return test_input


def datacleaning(text):
    porter_stem = PorterStemmer()
    stop = set(stopwords.words('english'))
    soup = BeautifulSoup(text)
    text = soup.get_text()
    token = re.split(r'\W+', text)
    token = [i.lower() for i in token]
    cleaned_sent = " ".join(token)
    token = re.split(r'\W+', cleaned_sent)
    token = [word for word in token if not word in stop]
    words = [porter_stem.stem(i) for i in token]
    cleaned_sent = " ".join(token)
    return cleaned_sent

def data_clean_test(test_input):
    df_preprocessed = test_input.copy()
    df_preprocessed['product_title'] = test_input['product_title'].apply(lambda x : datacleaning(x))
    df_preprocessed['product_brand'] = test_input['product_brand'].apply(lambda x : datacleaning(x))
    df_preprocessed['product_description'] = test_input['product_description'].apply(lambda x : datacleaning(x))
    df_preprocessed['search_term'] = test_input['search_term'].apply(lambda x : datacleaning(x))
    return df_preprocessed

def P(word, N=sum(WORDS.values())):
    "Probability of `word`."
    return WORDS[word] / N
def correction(word):
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)
def candidates(word):
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or set([word]))
def known(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)
def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)
def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))
def corrected_term(term):
  temp = term.lower().split()
  temp = [correction(word) for word in temp]
  return ' '.join(temp)

def search_correct(test_input):
    test_input['search_term_corrected'] = test_input['search_term'].apply(lambda x: corrected_term(x))
    test_input.drop(['search_term'],axis=1,inplace=True)
    return test_input


def common_words(df, column1, column2, cw_feature):
    w1_list = list(df[column1])
    w2_list = list(df[column2])
    cw = []
    for i in range(len(w1_list)):
        w1_list[i] = w1_list[i].lower().split()

    for i in range(len(w2_list)):
        w2_list[i] = w2_list[i].lower().split()
    for i in range(len(w1_list)):
        cw.append(len(set(w1_list[i]) & set(w2_list[i])))
    df[cw_feature] = cw
    return '1'


def word_counts(df, c1, c2, c3):
    w1_count = list(df[c1])
    w2_count = list(df[c2])
    w3_count = list(df[c3])
    for i in range(len(w1_count)):
        w1_count[i] = len(w1_count[i].lower().split())
        w2_count[i] = len(w2_count[i].lower().split())
        w3_count[i] = len(w3_count[i].lower().split())
    df['title_count'] = w1_count
    df['desc_count'] = w2_count
    df['seach_count'] = w3_count


# st_firstword, sb_firstword, st_lastword, sb_lastword
# from course videos quora problem
def first_last(search, title, brand):
    search_tokens = search.split()
    title_tokens = title.split()
    brand_tokens = brand.split()
    first_last_features = [0.0] * 4

    # First word of search and title
    if (len(search_tokens) == 0 or len(brand_tokens) == 0):
        first_last_features[0] = 0
        first_last_features[1] = 0
        first_last_features[2] = 0
        first_last_features[3] = 0
    else:
        first_last_features[0] = int(search_tokens[0] == title_tokens[0])

        # First word of search and brand
        first_last_features[1] = int(search_tokens[0] == brand_tokens[0])

        # Last word of search and title
        first_last_features[2] = int(search_tokens[-1] == title_tokens[-1])

        # Last word of search and brand
        first_last_features[3] = int(search_tokens[-1] == brand_tokens[-1])

    return first_last_features


# https://github.com/kriz17/Home-Depot-Product-Search-Relevance

def cosine_sim(t1, t2):
    text1 = set(t1.split())
    text2 = set(t2.split())
    den = math.sqrt(len(text1)) * math.sqrt(len(text2))
    num = len(text1 & text2)

    if not den:
        return 0.0
    else:
        return num / den


def jacquard_coeff(t1, t2):
    text1 = set(t1.split())
    text2 = set(t2.split())
    numerator = len(text1 & text2)
    denominator = len(text1 | text2)

    if not denominator:
        return 0.0
    else:
        return numerator / denominator


def numerical_features_test(test_input):
    common_words(test_input, 'product_title', 'product_description', 'title_desc_cw')
    common_words(test_input, 'search_term_corrected', 'product_title', 'search_title_cw')
    common_words(test_input, 'search_term_corrected', 'product_description', 'search_desc_cw')
    common_words(test_input, 'product_brand', 'search_term_corrected', 'brand_search_cw')
    common_words(test_input, 'product_brand', 'product_title', 'brand_title_cw')

    word_counts(test_input, 'product_title', 'product_description', 'search_term_corrected')
    test_input['title_freq'] = test_input.groupby('product_title')['product_title'].transform('count')
    test_input['searchterm_freq'] = test_input.groupby('search_term_corrected')['search_term_corrected'].transform(
        'count')
    test_input['word_total'] = test_input['desc_count'] + test_input['seach_count'] + test_input['title_count']
    test_input['search_desc_share'] = test_input['search_desc_cw'] / test_input['word_total']
    test_input['search_brand_share'] = test_input['brand_search_cw'] / test_input['word_total']
    test_input['search_title_share'] = test_input['search_title_cw'] / test_input['word_total']

    token_features_test = test_input.apply(
        lambda x: first_last(x['search_term_corrected'], x['product_title'], x['product_brand']), axis=1)
    test_input["first_word_st"] = list(map(lambda x: x[0], token_features_test))
    test_input["first_word_sb"] = list(map(lambda x: x[1], token_features_test))
    test_input["last_word_st"] = list(map(lambda x: x[2], token_features_test))
    test_input["last_word_sb"] = list(map(lambda x: x[3], token_features_test))

    # search and title
    test_input["token_set_ratio_st"] = test_input.apply(
        lambda x: fuzz.token_set_ratio(x["search_term_corrected"], x["product_title"]), axis=1)
    test_input["token_sort_ratio_st"] = test_input.apply(
        lambda x: fuzz.token_sort_ratio(x["search_term_corrected"], x["product_title"]), axis=1)
    test_input["fuzz_ratio_st"] = test_input.apply(
        lambda x: fuzz.QRatio(x["search_term_corrected"], x["product_title"]), axis=1)
    test_input["fuzz_partial_ratio_st"] = test_input.apply(
        lambda x: fuzz.partial_ratio(x["search_term_corrected"], x["product_title"]), axis=1)

    # search and brand
    test_input["token_set_ratio_sb"] = test_input.apply(
        lambda x: fuzz.token_set_ratio(x["search_term_corrected"], x["product_brand"]), axis=1)
    test_input["token_sort_ratio_sb"] = test_input.apply(
        lambda x: fuzz.token_sort_ratio(x["search_term_corrected"], x["product_brand"]), axis=1)
    test_input["fuzz_ratio_sb"] = test_input.apply(
        lambda x: fuzz.QRatio(x["search_term_corrected"], x["product_brand"]), axis=1)
    test_input["fuzz_partial_ratio_sb"] = test_input.apply(
        lambda x: fuzz.partial_ratio(x["search_term_corrected"], x["product_brand"]), axis=1)

    # search and description
    test_input["token_set_ratio_sd"] = test_input.apply(
        lambda x: fuzz.token_set_ratio(x["search_term_corrected"], x["product_description"]), axis=1)
    test_input["token_sort_ratio_sd"] = test_input.apply(
        lambda x: fuzz.token_sort_ratio(x["search_term_corrected"], x["product_description"]), axis=1)
    test_input["fuzz_ratio_sd"] = test_input.apply(
        lambda x: fuzz.QRatio(x["search_term_corrected"], x["product_description"]), axis=1)
    test_input["fuzz_partial_ratio_sd"] = test_input.apply(
        lambda x: fuzz.partial_ratio(x["search_term_corrected"], x["product_description"]), axis=1)

    # cosine coefficient
    test_input['cos_st'] = test_input.apply(lambda row: cosine_sim(row['search_term_corrected'], row['product_title']),
                                            axis=1)
    test_input['cos_sd'] = test_input.apply(
        lambda row: cosine_sim(row['search_term_corrected'], row['product_description']), axis=1)
    test_input['cos_sb'] = test_input.apply(lambda row: cosine_sim(row['search_term_corrected'], row['product_brand']),
                                            axis=1)

    # Jacquard coeff
    test_input['jac_st'] = test_input.apply(
        lambda row: jacquard_coeff(row['search_term_corrected'], row['product_title']), axis=1)
    test_input['jac_sd'] = test_input.apply(
        lambda row: jacquard_coeff(row['search_term_corrected'], row['product_description']), axis=1)
    test_input['jac_sb'] = test_input.apply(
        lambda row: jacquard_coeff(row['search_term_corrected'], row['product_brand']), axis=1)

    return test_input

def cosine_vec(v1, v2):
    num = np.dot(v1, v2)
    den = norm(v1)*norm(v2)
    if den != 0:
        return num/den
    else:
        return 0

def jacquard_vec(v1,v2 ):
    num = np.dot(v1,v2)
    den = norm(v1)**2 + norm(v2)**2 - np.dot(v1,v2)
    if den != 0:
        return num/den
    else:
        return 0

def cos_sim_apply(f1,f2):
    cosine = []
    for i in range(len(f1)):
        cosine.append(cosine_vec(f1[i], f2[i]))
    return cosine

def jaq_sim_apply(f1,f2):
    jaq = []
    for i in range(len(f1)):
        jaq.append(jacquard_vec(f1[i], f2[i]))
    return jaq

def bow_features(train_feature, bow_vectorizer):
    X_train_bow = bow_vectorizer.transform(train_feature.values)
    return X_train_bow

def bow_test(test_input):
    X_train_title_bow = bow_features(test_input['product_title'],bow_title)
    X_train_desc_bow = bow_features(test_input['product_description'],bow_title)
    X_train_brand_bow = bow_features(test_input['product_brand'],bow_title)
    X_train_search_bow = bow_features(test_input['search_term_corrected'],bow_title)

    arr_search = X_train_search_bow.toarray()
    list_search = arr_search.tolist()
    arr_search = []
    arr_title = X_train_title_bow.toarray()
    list_title = arr_title.tolist()
    arr_title = []
    arr_desc = X_train_desc_bow.toarray()
    list_desc = arr_desc.tolist()
    arr_desc = []

    test_input['cos_sim_bow_st'] = cos_sim_apply(list_title, list_search)
    test_input['cos_sim_bow_sd'] = cos_sim_apply(list_desc, list_search)

    test_input['jaq_sim_bow_st'] = jaq_sim_apply(list_title, list_search)
    test_input['jaq_sim_bow_sd'] = jaq_sim_apply(list_desc, list_search)

    return test_input

#avgw2v
def avgw2v_features(feature):
    avg_w2v = []; # final W2V
    for sentence in feature: # for each row
        vector = np.zeros(300) # initialize
        cnt_words =0; # num of words with a valid vector in the sentence
        for word in sentence.split(): # for each word in a sentence
            if word in glove_words:
                vector += model[word]
                cnt_words += 1
        if cnt_words != 0:
            vector /= cnt_words
        avg_w2v.append(vector)
    return avg_w2v

#for product_title column
#the sentence length is 300
def avgw2v(test_input):
    test_title_avgw2v = avgw2v_features(list(test_input['product_title'].values))
    test_desc_avgw2v = avgw2v_features(list(test_input['product_description'].values))
    test_brand_avgw2v = avgw2v_features(list(test_input['product_brand'].values))
    test_search_avgw2v = avgw2v_features(list(test_input['search_term_corrected'].values))

    test_input['cos_sim_w2v_st'] = cos_sim_apply(test_title_avgw2v, test_search_avgw2v)
    test_input['cos_sim_w2v_sd'] = cos_sim_apply(test_desc_avgw2v, test_search_avgw2v)
    test_input['cos_sim_w2v_brand'] = cos_sim_apply(test_brand_avgw2v, test_search_avgw2v)

    test_input['jaq_sim_w2v_st'] = jaq_sim_apply(test_title_avgw2v, test_search_avgw2v)
    test_input['jaq_sim_w2v_sd'] = jaq_sim_apply(test_desc_avgw2v, test_search_avgw2v)
    test_input['jaq_sim_w2v_brand'] = jaq_sim_apply(test_desc_avgw2v, test_search_avgw2v)


    test_title_avgw2v = pd.DataFrame(test_title_avgw2v)
    test_desc_avgw2v= pd.DataFrame(test_desc_avgw2v)
    test_brand_avgw2v=pd.DataFrame(test_brand_avgw2v)
    test_search_avgw2v =pd.DataFrame(test_search_avgw2v)

    s4_train = pd.concat([test_title_avgw2v, test_desc_avgw2v, test_brand_avgw2v, test_search_avgw2v], axis=1)

    return s4_train,test_input

# TFIDF average Word2Vec on trian data
#Reffered from course assignments to implement W2V

def tfidf_avgw2v_features(feature,tfidfw2v):
    tfidf_w2v = []; #TFIDF W2V is stored in this list
    dictionary = tfidfw2v['dictionary']
    tfidf_words = tfidfw2v['tfidf_words']
    for sentence in feature:
        vector = np.zeros(300) # 300 features
        tf_idf_weight =0; # num of words with a valid vector in the sentence
        for word in sentence.split():
            if (word in glove_words) and (word in tfidf_words):
                vec = model[word]
                tf_idf = dictionary[word]*(sentence.count(word)/len(sentence.split())) # getting the tfidf value for each word
                vector += (vec * tf_idf) # calculating tfidf weighted w2v
                tf_idf_weight += tf_idf
        if tf_idf_weight != 0:
            vector /= tf_idf_weight
        tfidf_w2v.append(vector)
    return tfidf_w2v

def tfidfavgw2v(test_input):
    X_train_title_tfidfavgw2v = tfidf_avgw2v_features(list(test_input['product_title'].values),tfidfw2v_title)
    X_train_desc_tfidfavgw2v = tfidf_avgw2v_features(list(test_input['product_description'].values),tfidfw2v_desc)
    X_train_brand_tfidfavgw2v = tfidf_avgw2v_features(list(test_input['product_brand'].values),tfidfw2v_brand)
    X_train_search_tfidfavgw2v = tfidf_avgw2v_features(list(test_input['search_term_corrected'].values),tfidfw2v_search)

    test_input['cos_sim_tfidfw2v_st'] = cos_sim_apply(X_train_title_tfidfavgw2v, X_train_search_tfidfavgw2v)
    test_input['cos_sim_tfidfw2v_sd'] = cos_sim_apply(X_train_desc_tfidfavgw2v, X_train_search_tfidfavgw2v)
    test_input['cos_sim_tfidfw2v_brand'] = cos_sim_apply(X_train_brand_tfidfavgw2v, X_train_search_tfidfavgw2v)

    test_input['jaq_sim_tfidfw2v_st'] = jaq_sim_apply(X_train_title_tfidfavgw2v, X_train_search_tfidfavgw2v)
    test_input['jaq_sim_tfidfw2v_sd'] = jaq_sim_apply(X_train_desc_tfidfavgw2v, X_train_search_tfidfavgw2v)
    test_input['jaq_sim_tfidfw2v_brand'] = jaq_sim_apply(X_train_brand_tfidfavgw2v, X_train_search_tfidfavgw2v)

    X_train_title_tfidfavgw2v = pd.DataFrame(X_train_title_tfidfavgw2v)
    X_train_desc_tfidfavgw2v = pd.DataFrame(X_train_desc_tfidfavgw2v)
    X_train_brand_tfidfavgw2v =pd.DataFrame(X_train_brand_tfidfavgw2v)
    X_train_search_tfidfavgw2v =pd.DataFrame(X_train_search_tfidfavgw2v)

    s5_train = pd.concat([X_train_title_tfidfavgw2v, X_train_desc_tfidfavgw2v, X_train_brand_tfidfavgw2v, X_train_search_tfidfavgw2v], axis=1)

    return s5_train,test_input

def predict_test(test_example):
    try:
        test_input = test(test_example)
        print('1')
        test_input = merge_test(test_input)
        print('2')
        test_input = null_check(test_input)
        print('3')
        test_input = data_clean_test(test_input)
        print('4')
        test_input = search_correct(test_input)
        print('5')
        test_input = numerical_features_test(test_input)
        print('6')
        test_input = bow_test(test_input)
        print('7')
        avg_w2v = avgw2v(test_input)
        print('8')
        s4_train = avg_w2v[0]
        test_input = avg_w2v[1]
        tfidf_avg_w2v = tfidfavgw2v(test_input)
        print('9')
        s5_train = tfidf_avg_w2v[0]
        test_input = tfidf_avg_w2v[1]

        s5_train = s5_train.iloc[:, 0:1200]
        s4_train = s4_train.iloc[:, 0:1200]

        s1_train = test_input.copy()
        titles = s1_train['product_title'].values
        features_to_exclude = ['brand_search_cw', 'first_word_st', 'first_word_sb', 'last_word_st',
                               'last_word_sb', 'token_set_ratio_sd', 'token_sort_ratio_sd', 'fuzz_ratio_sd', 'cos_st',
                               'cos_sb', 'jac_st','jac_sb', 'cos_sim_bow_st', 'cos_sim_bow_sd',
                               'jaq_sim_bow_st', 'jaq_sim_bow_sd', 'search_title_cw',
                               'search_desc_cw', 'brand_title_cw','product_title', 'product_uid',
                               'product_description', 'product_brand', 'search_term_corrected']
        s1_train.drop(features_to_exclude,
                       axis=1, inplace=True)
        s4_train.reset_index(drop=True, inplace=True)
        s5_train.reset_index(drop=True, inplace=True)
        s1_train.reset_index(drop=True, inplace=True)
        final_features_test = pd.concat([s4_train,s5_train,s1_train],axis=1)
        

        m1 = RF.predict(final_features_test)
        m2 = GBDT.predict(final_features_test)
        c=[]
        for i in range(0,2433):
            c.append(i)
        final_features_test.columns = c
        m3 = XGB.predict(final_features_test)

        model_test = pd.DataFrame()
        model_test['title'] = titles
        model_test['m1'] = m1
        model_test['m2'] = m2
        model_test['m3'] = m3

        model_test['avg'] = (model_test['m1']
                                              + model_test['m2'] + model_test['m3']) / 3
        score = model_test['avg'].values
        return score, model_test
    except Exception as e:
        print(str(e))
        return "Sorry, the title is not listed in the Database."


