import tensorflow as tf
import csv
import numpy as np
import gensim
import gensim.downloader as api
from gensim.test.utils import common_texts
from gensim.test.utils import common_texts
from gensim.models import word2vec
from gensim.models import Word2Vec
#import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional
#from sklearn.model_selection import train_test_split
#from tensorflow.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
vocab_size=10000
vector_size=50
length=0
sentence_length=0
embedding_weights = ''
embedding_dims = 50
max_len = 300
vocab_len = 0

def RetDataset():
    with open('plot_tok_gt9_5000.csv', 'r', encoding='ISO-8859-1', newline='\n') as objective_data:
        objective_list = np.array(objective_data.readlines())#objective_list = (csv.reader(objective_data, delimiter=' ', skipinitialspace=False))

    with open('quote_tok_gt9_5000.csv', 'r', encoding='ISO-8859-1', newline='\n') as subjective_data:
        subjective_list = np.array(subjective_data.readlines())#subjective_list = (csv.reader(subjective_data, delimiter=' ', skipinitialspace=False))

    obj_size = 5000
    subj_size = 5000
    total_size = (obj_size + subj_size)
    training_categories = [''] * 10000
    training_sentences = [''] * 10000
    testing_sentences = [''] * 1000
    testing_categories = [''] * 1000
    longest_string = 0

    j = 0
    vector_sentences= list()
        #adds/mixes both datasets to the list of sentences
    for i in range(0, obj_size):
        training_categories[j] = 1  # 1 =  objective sentence
        training_sentences[j] = np.str.split(objective_list[i])
        longest_string=RetLongestStr(training_sentences[j], longest_string)
        for each in training_sentences[j]:
            vector_sentences.append(each)
        training_categories[j + 1] = 0  #0 = subjective sentence
        training_sentences[j + 1] = np.str.split(subjective_list[i])
        longest_string=RetLongestStr(training_sentences[j + 1], longest_string)
        for each in training_sentences[j + 1]:
            vector_sentences.append(each)
        j = j+2

    objective_data.close()
    subjective_data.close()

    #produces embedding vectors for processing
    training_sentences = Word2VecInit(vector_sentences, training_sentences)

    #attempts to pad the sequences
    print("start padding")
    max_len=longest_string
    training_sentences=pad_sequences(training_sentences, max_len, float)

    print("start randomize")
    # randomizes data to be placed into a 1000 length testing dataset
    c = list(zip(training_sentences, training_categories))
    np.random.shuffle(c)
    training_sentences, training_categories = zip(*c)

    #splits training data into training and testing data
    ((training_sentences, training_categories), (testing_sentences, testing_categories)) = \
        SplitTestData1(training_sentences, training_categories)

    print("end randomize test data")
    training_categories=to_categorical(training_categories)
    testing_categories=to_categorical(testing_categories)
    print("return data complete")
    training_sentences = np.reshape(training_sentences, (9000, 120, embedding_dims))
    testing_sentences = np.reshape(testing_sentences, (1000, 120, embedding_dims))
    return ((training_sentences, training_categories),(testing_sentences, testing_categories))

def RetLongestStr(str_list, longest_str):
    str_len=len(str_list)
    if(longest_str < str_len):
        return str_len
    else:
        return longest_str

def SplitTestData1(train_sen, train_cat):
    testing_length = len(train_sen)
    test_sen = train_sen[9000: testing_length]
    train_sen = train_sen[0:9000]
    test_cat = train_cat[9000: testing_length]
    train_cat = train_cat[0:9000]
    return ((train_sen, train_cat),(test_sen, test_cat))

def SplitTestData(train_sen, train_cat):
    testing_length = len(train_sen)
    print(testing_length)
    test_sen = train_sen[1800: testing_length]
    train_sen = train_sen[0:1800]
    test_cat = train_cat[1800: testing_length]
    train_cat = train_cat[0:1800]

    return ((train_sen, train_cat),(test_sen, test_cat))

def TrainSentData():
    with open('Combined_News_DJIA.csv', 'r', newline='') as DJIA_news_file:
        stock_newsdata = list(csv.reader(DJIA_news_file, delimiter=',', skipinitialspace=True))
        DJIA_news_file.close()
    with open('RedditNews.csv', 'r', newline='') as DJIA_news_file2:
         reddit_newsdata = list(csv.reader(DJIA_news_file2, delimiter=',', skipinitialspace=True))
         DJIA_news_file2.close()
    newsdata_targets=[]
    dates=[]
    newsdata=[]
    text=[]
    newsdata_targets.append(0)
    for i in range(1, len(stock_newsdata)):
        newsdata_targets.append(stock_newsdata[i][1])
        dates.append([np.char.replace(stock_newsdata[i][0], '-', '')])
        #newsdata.append([np.str.split(stock_newsdata[i][2])])
        for j in range(2, len(stock_newsdata[i])):
            #print(stock_newsdata[i][j])
            #newsdata.append([np.str.split(stock_newsdata[i][j])])
            #stock_newsdata[i][j]=np.str.split(stock_newsdata[i][j])
            stock_newsdata[i][j]=np.str.split(stock_newsdata[i][j])
            newsdata.append(stock_newsdata[i][j])

    newsdata, sentence_length=WordVectors(newsdata, stock_newsdata)

    sc=MinMaxScaler(feature_range=(0, 1))
    dates=sc.fit_transform(dates)
    print('start conc')
    data=[]
    cat_data=[]
    #print(newsdata)
    k=0
    for i in range(1, len(stock_newsdata)):
        if(k<len(newsdata)):
            length=len(stock_newsdata[i])
            stock_newsdata[i][0] = ([np.array(
                [np.char.replace(stock_newsdata[i][0], '-', ''), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0], dtype=np.float)])

        #for j in range(2, (length)):
         #   if(k<len(newsdata)):

        #        data[i-1].extend(newsdata[k])
                #cat_data.append(newsdata_targets[i])
        #        k+=1

    print('Start padding')
    stock_newsdata[0]=stock_newsdata[1]
    #data.append(data[len(data) - 1])
    data=pad_sequences(stock_newsdata, sentence_length, dtype=object)
    print('Start randomization')
    c = list(zip(data, newsdata_targets))
    np.random.shuffle(c)
    (stock_newsdata, newsdata_targets) = zip(*c)

    ((training_sentences, training_categories), (testing_sentences, testing_categories))=\
        SplitTestData(stock_newsdata, newsdata_targets)
    print('End Data-Processing')
    return (training_sentences, training_categories), (testing_sentences, testing_categories)

def Word2VecInit(word_list, words):
    vocab_len=len(word_list)
    model = Word2Vec(sentences=words, size=embedding_dims,  window=3, min_count=1, workers=3)
    try:
        model.load("word2vec.model")
        print('word2vec loaded')
    except:
        model.train(words, total_words=vocab_len, epochs=10)
        model.save("word2vec.model")
        print('word2vec trained')
    #model.load("word2vec.model", map='r')
    #word_vectors=model.wv
    vector_words=[]
    for i in range(0, len(words)):
        vector_words.append([])
        for j in range(0, len(words[i])):
           vector_words[i].append(model.wv[words[i][j]])
        #print(vector_words[i])
    print("wordvec complete")
            #word_list[i]=word_vectors(word_list[i][j])
    return vector_words

def WordVectors(sentences, sentiment):
    print('start vector processing')
    #texts = sentences[0:len(sentences)][1:len(sentences[0])]
    #model = api.load('glove-twitter-25')
    model = word2vec.Word2Vec(sentences=sentences, workers=4, size=vector_size, window=3, min_count=1)
    #model.build_vocab(texts)
    model.train(sentences, total_examples=len(sentences), epochs=1)
    model.save('news_data')
    vectors=[]


    vectors=[]
    max_len=0
    for i in range(1, len(sentiment)):
        #print(i)
        in_list=[]
        if(max_len<len(sentiment[i])):
            max_len=len(sentiment[i])
        for j in range(1,len(sentiment[i])):
            #print(each)
            #print(model.wv[each])
            sentiment[i][j] = (model.wv[sentiment[i][j]])
    print('end')
    return sentiment, max_len

def TrainModel(texts, targets, test_data, test_targets):
    texts=np.array(texts)
    test_data=np.array(test_data)
    test_data=np.reshape(testdata, (1990, 27, 1), dtype=float)
    targets=to_categorical(targets)
    print(test_data)
    test_targets=to_categorical(test_targets)
    data_length=len(texts[0])
    sentence_length=len(texts[0][0])
    print(len(texts[0][0]))
    print(data_length)
    print(sentence_length)
    sentence_length
    SELSTM = keras.models.Sequential()
    #SELSTM.add(keras.layers.Embedding(100000, 27))
    #SELSTM.add(layers.Conv1D(128, 8, activation='relu'))
    #SELSTM.add(layers.Reshape((19, 2304)))
    #SELSTM.add(layers.MaxPooling2D(18, 4))
    #SELSTM.add(layers.Dropout(0.1))
    SELSTM.add(keras.layers.LSTM(128,  return_sequences=True))
    #SELSTM.add(layers.Flatten())
    #SELSTM.add(layers.Conv1D(32, 8, activation='tanh'))
    #SELSTM.add(layers.MaxPooling1D(8,4))
    #SELSTM.add(layers.Dropout(0.1))
    #SELSTM.add(layers.MaxPooling1D(32, 16))
    SELSTM.add(layers.LSTM(32 ))
    #SELSTM.add(layers.MaxPooling1D(32, 16))
    SELSTM.add(layers.Flatten())
    SELSTM.add(layers.Dense(32, activation='relu'))
    SELSTM.add(layers.Dense(2, activation='softmax'))
    SELSTM.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    SELSTM.fit(texts, targets,
              epochs=15,
              batch_size=180,
               validation_split=0.1)
    SELSTM.evaluate(test_data, test_targets)

(training_sentences, training_categories), (testing_sentences, testing_categories)=TrainSentData()#
#((training_sentences, training_categories), (testing_sentences, testing_categories))=RetDataset()
TrainModel(training_sentences, training_categories, testing_sentences, testing_categories)
