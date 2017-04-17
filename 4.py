#encoding=utf-8
import time,datetime
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

import math
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import word2vec

import sys
sys.path.append("/Users/xogo/Desktop/NTU/2017 spring/NLP/project1/NLP_opinion_mining/")
sys.path.append("/Users/xogo/Desktop/NTU/2017 spring/NLP/project1/NLP_opinion_mining/data/")



def Review_to_vec_array(train,test,max_comma) :
    review_train = np.array(train.loc[:,'Review'])
    aspect_vec_train = np.array(train.loc[:,'aspect_vec'])
    review_test = np.array(test.loc[:,'Review'])
    review = np.append(review_train,review_test,axis=0)

    #lst = []
    lst_CountVec = []
    lst_train_omc = [] #omc means over max comma
    lst_test_omc = []
    num_train_omc = 0
    num_test_omc = 0
    lst_avt = [] # avt means aspect_vec_train
    lst_avtest = []
    index = 0
    for row in review :
        lst_temp = row.split(',')
        #lst_CountVec = lst_CountVec + lst_temp
        if len(lst_temp) <= max_comma :
            #lst = lst + lst_temp
            lst_CountVec = lst_CountVec + lst_temp
            if index < 200 :
                lst_avt.append([int(i) for i in aspect_vec_train[index].split(' ') ])
        else :
            if index < 200 :
                lst_train_omc = lst_train_omc + [index]
                num_train_omc = num_train_omc + 1
            else :
                lst_test_omc = lst_test_omc + [index-200]
                num_test_omc = num_test_omc + 1
        index = index + 1
    #m = max(l)

    vectorizer=CountVectorizer(min_df=1,ngram_range=(1,1),stop_words=None,analyzer=u'char')#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    vec_ary = vectorizer.fit_transform(lst_CountVec)

    vec_train = vec_ary[:(200-num_train_omc)*max_comma][:].toarray()
    vec_test = vec_ary[(200-num_train_omc)*max_comma:][:].toarray()

    X_train = np.reshape(vec_train, (int(vec_train.shape[0]/max_comma),max_comma,vec_train.shape[1]))
    X_test = np.reshape(vec_test, (int(vec_test.shape[0]/max_comma),max_comma,vec_test.shape[1]))

    Y_train = lst_avt

    return (X_train,Y_train,X_test)

#服務, 環境, 價格, 交通, 餐廳
def aspect_to_vec(ary) :
    #ary['aspect_vec'] = pd.DataFrame(0, index=np.arange(len(ary)))
    ary = ary.fillna('') # exchange NaN into ''
    for row in range(len(ary)) :
        vec = np.zeros((15,), dtype=np.int)
        if '服務' in ary.loc[row,'Neg'] :
            vec[0] = 1
        elif '服務' in ary.loc[row,'Pos'] :
            vec[2] = 1
        else :
            vec[1] = 1
        if '環境' in ary.loc[row,'Neg'] :
            vec[3] = 1
        elif '環境' in ary.loc[row,'Pos'] :
            vec[5] = 1
        else :
            vec[4] = 1
        if '價格' in ary.loc[row,'Neg'] :
            vec[6] = 1
        elif '價格' in ary.loc[row,'Pos'] :
            vec[8] = 1
        else :
            vec[7] = 1
        if '交通' in ary.loc[row,'Neg'] :
            vec[9] = 1
        elif '交通' in ary.loc[row,'Pos'] :
            vec[11] = 1
        else :
            vec[10] = 1
        if '餐廳' in ary.loc[row,'Neg'] :
            vec[12] = 1
        elif '餐廳' in ary.loc[row,'Pos'] :
            vec[14] = 1
        else :
            vec[13] = 1
        s = ' '.join(str(ff) for ff in vec)
        ary.loc[row,'aspect_vec'] = s

    return ary


def comma_reset(ary,num_max) :
    #train['aspect_vec'] = pd.DataFrame(0, index=np.arange(len(train)))
    ary = ary.fillna('') # exchange NaN into ''
    for row in range(len(ary)) :

        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('。',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('，',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('！',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('!',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('？',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('?',',')
        #ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('、',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('\'','')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('  ',' ')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace(', ,',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace(',,',',')
        '''
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('（',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('）',',')
        '''

        num_comma = ary.loc[row,'Review'].count(',')
        t1 = num_comma
        t2 = num_comma
        while t1 < num_max-1 :
            #ary.loc[row,'Review'] = ary.loc[row,'Review'] + ','
            ary.loc[row,'Review'] = ',' + ary.loc[row,'Review']
            t1 = t1 + 1
        while t2 > num_max-1 :
            p1 = 0
            p2 = 0
            for i in range(int(t2/2)) :
                p1 = ary.loc[row,'Review'].find(',' , p2)
                p2 = ary.loc[row,'Review'].find(',' , p1)
                ary.loc[row,'Review'] = ary.loc[row,'Review'][:p2] + ary.loc[row,'Review'][p2+1:]
                t2 = t2 - 1
                if t2 <= num_max - 1 :
                    break
            #t2 = int(t2/2) + 1
    return ary


#1.服務, 2.環境, 3.價格, 4.交通, 5.餐廳
def Y_select(Y_train,type_Y) :
    i1 = type_Y * 3 -3
    i2 = type_Y * 3 -1
    ary = []
    for row in Y_train :
        if row[i1] != 0 or row[i2] != 0 :
            ary.append(row[i1])
            ary.append(row[i2])
        else :
            ary.append(0.5)
            ary.append(0.5)
    ary = np.asarray(ary)
    ary = ary.reshape((len(Y_train),2))
    return ary


def create_model(X_train,max_comma) :
    model = Sequential()

    #model.add(LSTM(64, input_dim=1801, input_length=20, return_sequences=True))
    model.add(LSTM(32, input_dim=X_train.shape[2], input_length=max_comma)) # 注意 lstm的輸入和輸出，timestep是指要回傳自己幾次
    #model.add(LSTM(32, return_sequences=Tru))
    #model.add(LSTM(32))

    #model.add(Dense(350, init='uniform', activation='relu')) # try softmax
    #model.add(Dense(100, init='uniform', activation='relu')) # try softmax
    #model.add(Dense(30, init='uniform', activation='relu')) # try softmax
    model.add(Dense(2, init='uniform', activation='softmax')) # try softmax
    #model.compile(loss='mean_squared_error', optimizer='adam')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model




def test_threshold(Y_train1,predictions) :
    score = []
    ans_all = []
    for i in range(0,100) :
        ans=[]
        th = i/100.0
        (s,ans) = set_new_predictions(th,predictions)
        score.append(s)
        ans_all.append(ans)

    return score


def set_new_predictions(th,predictions,Y_train_n) : # th means threshold
    ans = []
    s = 0
    index = 0
    ans2 = []
    if Y_train_n == None :
        for j in predictions :
            if j[0] - j[1] > th :
                ary=[1,0]
                ans.append(ary)
                ans2.append(-1)
                index = index + 1
            elif j[1] - j[0] > th :
                ary=[0,1]
                ans.append(ary)
                ans2.append(1)
                index = index + 1
            else :
                ary = [0,0]
                ans.append(ary)
                ans2.append(0)
                index = index + 1
    else :
        for j in predictions :
            if j[0] - j[1] > th :
                ary=[1,0]
                ans.append(ary)
                ans2.append(-1)
                if Y_train_n[index].tolist() == ary :
                    s = s + 1
                index = index + 1
            elif j[1] - j[0] > th :
                ary=[0,1]
                ans.append(ary)
                ans2.append(1)
                if Y_train_n[index].tolist() == ary :
                    s = s + 1
                index = index + 1
            else :
                ary = [0,0]
                ans.append(ary)
                ans2.append(0)
                if Y_train_n[index].tolist() == ary :
                    s = s + 1
                index = index + 1

    return (s, ans, ans2)


def w2v(sentence,save=False) :
    aspect_review = pd.read_csv('data/aspect_review_seg.csv')
    polarity_review = pd.read_csv('data/polarity_review_seg.csv')
    test_review = pd.read_csv('data/test_seg.csv')
    test_review = test_review.groupby('Review_id').first()
    reviews = aspect_review['Review'].append(polarity_review['Review']).append(test_review['Review'])
    print('#reviews:', len(reviews))

    if os.path.exists('data/external_corpus.txt'):
        print('using extrenal corpus, it may take pretty loooong')
        with open('data/external_corpus.txt', "r") as inputs:
            with open('data/extrenal_corpus.txt', 'w') as cutted:
                for line in inputs.readlines():
                    cutted.write(jieba.cut(line))
    else:
        print('external corpus dosen\'t exitst (put it as data/external_corpus.txt plz QAQ)')


    with open("data/word2vec_corpus.tmp", "w") as f:
        if os.path.exists('data/external_corpus_seg.txt'):
            with open("data/external_corpus_seg.txt") as ext:
                f.write(ext.read())
            f.write(("\n".join(reviews)+"\n")*5)
        else:
            f.write(("\n".join(reviews)))

    if os.path.exists('word2vec/word2vec'):
        print('running word2vec ...')
        os.system('time word2vec/word2vec -train data/word2vec_corpus.tmp -output vectors.bin -cbow 1 -size 300 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15')
    else:
        print('word2vec doesn\'t exitst (put it as word2vec/word2vec plz QAQ)')

if __name__=='__main__' :
    print ("start time : " ,datetime.datetime.now())
    start_time = time.time()


    max_comma = 20
    th = 0.8 #threshold
    epochs = 20
    train = pd.read_csv("./data/aspect_review_seg.csv")#,header=None,index_col=False)
    train = aspect_to_vec(train)
    train = comma_reset(train,max_comma)
    test = pd.read_csv("./data/test_seg.csv")#,header=None,index_col=False)
    test = comma_reset(test,max_comma)
    question = pd.read_csv("./data/test.csv")
    ans = pd.read_csv("./data/sample_submission.csv")


    (X_train,Y_train,X_test) = Review_to_vec_array(train,test,max_comma)


    Y_train_1 = Y_select(Y_train,1)
    Y_train_2 = Y_select(Y_train,2)
    Y_train_3 = Y_select(Y_train,3)
    Y_train_4 = Y_select(Y_train,4)
    Y_train_5 = Y_select(Y_train,5)

    model1 = create_model(X_train)
    model2 = create_model(X_train)
    model3 = create_model(X_train)
    model4 = create_model(X_train)
    model5 = create_model(X_train)

    model1.fit(X_train, Y_train_1, epochs=epochs, verbose=2)
    model2.fit(X_train, Y_train_2, epochs=epochs, verbose=2)
    model3.fit(X_train, Y_train_3, epochs=epochs, verbose=2)
    model4.fit(X_train, Y_train_4, epochs=epochs, verbose=2)
    model5.fit(X_train, Y_train_5, epochs=epochs, verbose=2)


    predictions1 = model1.predict(X_test)
    predictions2 = model2.predict(X_test)
    predictions3 = model3.predict(X_test)
    predictions4 = model4.predict(X_test)
    predictions5 = model5.predict(X_test)

    (s1, ans_1, ans1) = set_new_predictions(th,predictions1,None)
    (s2, ans_2, ans2) = set_new_predictions(th,predictions2,None)
    (s3, ans_3, ans3) = set_new_predictions(th,predictions3,None)
    (s4, ans_4, ans4) = set_new_predictions(th,predictions4,None)
    (s5, ans_5, ans5) = set_new_predictions(th,predictions5,None)

    '''
    predictions1 = model1.predict(X_train)
    predictions2 = model2.predict(X_train)
    predictions3 = model3.predict(X_train)
    predictions4 = model4.predict(X_train)
    predictions5 = model5.predict(X_train)

    (s1, ans_1, ans1) = set_new_predictions(th,predictions1,Y_train_1)
    (s2, ans_2, ans2) = set_new_predictions(th,predictions2,Y_train_2)
    (s3, ans_3, ans3) = set_new_predictions(th,predictions3,Y_train_3)
    (s4, ans_4, ans4) = set_new_predictions(th,predictions4,Y_train_4)
    (s5, ans_5, ans5) = set_new_predictions(th,predictions5,Y_train_5)
    '''

    #scores = model.evaluate(X_train, Y_train1)
    #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



    item = ['服務', '環境', '價格', '交通', '餐廳']
    #1.服務, 2.環境, 3.價格, 4.交通, 5.餐廳
    for i in range(len(question)) :
        if question['Aspect'][i] == item[0] :
            ans['Label'][i] = ans1[i]
        elif question['Aspect'][i] == item[1] :
            ans['Label'][i] = ans2[i]
        elif question['Aspect'][i] == item[2] :
            ans['Label'][i] = ans3[i]
        elif question['Aspect'][i] == item[3] :
            ans['Label'][i] = ans4[i]
        elif question['Aspect'][i] == item[4] :
            ans['Label'][i] = ans5[i]


    ans.to_csv('ans_lstm_20comma_32dim_20epochs_th0.8_0.5-0.5.csv',index=False)

    print ("finish time : " , datetime.datetime.now())
    print ("running time is %s seconds" %(time.time()-start_time) )




    '''
    # create and fit the LSTM network
    model = Sequential()
    #model.add(LSTM(output_dim, batch_input_shape=(batch_size, timesteps, data_dim)))
    model.add(LSTM(64, batch_input_shape=(1,X_train.shape[1],X_train.shape[2])))
    model.add(Dense(15, init='uniform', activation='linear')) # try softmax
    #model.compile(loss='mean_squared_error', optimizer='adam')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=1, batch_size=1, verbose=2)

    scores = model.evaluate(X_train, Y_train, batch_size=1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    predictions = model.predict(X_test)
    '''



    '''
    # LSTM and CNN for sequence classification in the IMDB dataset
    import numpy
    from keras.datasets import imdb
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers.convolutional import Conv1D
    from keras.layers.convolutional import MaxPooling1D
    from keras.layers.embeddings import Embedding
    from keras.preprocessing import sequence
    # fix random seed for reproducibility
    numpy.random.seed(7)
    # load the dataset but only keep the top n words, zero the rest
    top_words = 5000
    (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)
    # truncate and pad input sequences
    max_review_length = 500
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
    # create the model
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, epochs=3, batch_size=64)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    '''

    '''
    # LSTM for sequence classification in the IMDB dataset
    import numpy
    from keras.datasets import imdb
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers.embeddings import Embedding
    from keras.preprocessing import sequence
    # fix random seed for reproducibility
    numpy.random.seed(7)
    # load the dataset but only keep the top n words, zero the rest
    top_words = 5000
    (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)
    # truncate and pad input sequences
    max_review_length = 500
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
    # create the model
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, nb_epoch=3, batch_size=64)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    '''
