#encoding=utf-8
import os
import time,datetime
import pandas as pd
import numpy as np
import keras
from keras.layers import LSTM, Embedding, Input, Dense, TimeDistributed, Activation
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
import word2vec

import sys
sys.path.append("/Users/xogo/Desktop/NTU/2017 spring/NLP/project1/NLP_opinion_mining/")
sys.path.append("/Users/xogo/Desktop/NTU/2017 spring/NLP/project1/NLP_opinion_mining/data/")


#服務, 環境, 價格, 交通, 餐廳
def aspect_to_vec(ary) :
    #ary['aspect_vec'] = pd.DataFrame(0, index=np.arange(len(ary)))
    ary = ary.fillna('') # exchange NaN into ''
    for row in range(len(ary)) :
        vec = np.zeros((5,), dtype=float)
        if '服務' in ary.loc[row,'Neg'] :
            vec[0] = 0.0
        elif '服務' in ary.loc[row,'Pos'] :
            vec[0] = 1.0
        else :
            vec[0] = 0.5
        if '環境' in ary.loc[row,'Neg'] :
            vec[1] = 0.0
        elif '環境' in ary.loc[row,'Pos'] :
            vec[1] = 1.0
        else :
            vec[1] = 0.5
        if '價格' in ary.loc[row,'Neg'] :
            vec[2] = 0.0
        elif '價格' in ary.loc[row,'Pos'] :
            vec[2] = 1.0
        else :
            vec[2] = 0.5
        if '交通' in ary.loc[row,'Neg'] :
            vec[3] = 0.0
        elif '交通' in ary.loc[row,'Pos'] :
            vec[3] = 1.0
        else :
            vec[3] = 0.5
        if '餐廳' in ary.loc[row,'Neg'] :
            vec[4] = 0.0
        elif '餐廳' in ary.loc[row,'Pos'] :
            vec[4] = 1.0
        else :
            vec[4] = 0.5
        s = ' '.join(str(ff) for ff in vec)
        ary.loc[row,'aspect_vec'] = s

    return ary

def comma_reset(ary,num_max) :
    #train['aspect_vec'] = pd.DataFrame(0, index=np.arange(len(train)))
    ary = ary.fillna('') # exchange NaN into ''
    for row in range(len(ary)) :
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('。',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('！',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('!',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('？',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('?',',')
        #ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('、',',')
        #ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('\'','')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('。',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('\n',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('，',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('~',' ')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('～',' ')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace(';',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('   ',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('   ',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('  ',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('  ',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('.',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace(' ,',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace(', ',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace(' ,',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace(', ',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace(', ,',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace(', ,',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace(',,',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace(',,',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace(',,',',')
        #ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('','')
        '''
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('（',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('）',',')
        '''


        num_comma = ary.loc[row,'Review'].count(',')
        t1 = num_comma
        t2 = num_comma
        flag = 0
        while t1 < num_max-1 :
            #ary.loc[row,'Review'] = ary.loc[row,'Review'] + ','
            ary.loc[row,'Review'] = ',' + ary.loc[row,'Review']
            t1 = t1 + 1
        while t2 > num_max-1 :
            p1 = 0
            p2 = 0
            for i in range(int(t2/2)) :
                p1 = ary.loc[row,'Review'].find(',' , p2+1)
                p2 = ary.loc[row,'Review'].find(',' , p1+1)
                if p2 - p1 > 20 and flag < 20:
                    flag += 1
                    continue
                if p2 == -1 or p1 == -1:
                    p2 = 0
                    continue
                ary.loc[row,'Review'] = ary.loc[row,'Review'][:p2] +' '+ ary.loc[row,'Review'][p2+1:]
                t2 = t2 - 1
                if t2 <= num_max - 1 :
                    break
            #t2 = int(t2/2) + 1
    return ary

def w2v(w2v_dim=300, aspect_review=None, test_review=None) :
    if aspect_review is None :
        print('nooooooo')
        return 0
    polarity_review = pd.read_csv('data/polarity_review_seg.csv')
    test_review = test_review.groupby('Review_id').first()
    reviews = aspect_review['Review'].append(polarity_review['Review']).append(test_review['Review'])
    print('#reviews:', len(reviews))

    with open("data/word2vec_corpus.tmp", "w") as f:
            f.write(("\n".join(reviews)+"\n"))
    print('running word2vec ...')
    word2vec.word2phrase('data/word2vec_corpus.tmp', 'data/word2vec_corpus_phrases', verbose=True)
    word2vec.word2vec('data/word2vec_corpus_phrases', 'data/word2vec_corpus.bin', size=w2v_dim, verbose=True, window=5, cbow=0, binary=1, min_count=1, sample='1e-5', hs=1, iter_=5)
#0.服務, 1.環境, 2.價格, 3.交通, 4.餐廳
def Y_select(Y_train,type_Y) :
    lst = []
    index = 0
    for row in Y_train :
        Y_train_lst = row.split(' ')
        lst.append(float(Y_train_lst[type_Y]))
        index += 1
    ary = np.asarray(lst)
    return ary

def Review_to_vec(train, test, model_w2v, w2v_dim=300, max_comma=20, len_sentence=65) :
    review_train = np.array(train.loc[:,'Review'])
    review_test = np.array(test.loc[:,'Review'])
    review = np.append(review_train,review_test,axis=0)

    ary_train = np.zeros((200,max_comma,len_sentence,w2v_dim))
    ary_test = np.zeros((1737,max_comma,len_sentence,w2v_dim))
    i1 = 0
    for row in review :
        #print(i1)
        if i1 < 200 :
            lst_temp = row.split(',')
            i2 = 0
            for sentence in lst_temp :
                lst_temp2 = sentence.split(' ')
                i3 = 0
                if lst_temp2 != [''] :
                    '''
                    if len(lst_temp2) > 50 :
                        print(i1)
                        print(len(lst_temp2))
                        print (row)
                        #print (lst_temp)
                        #print (lst_temp2)
                    '''
                    for word in lst_temp2 :
                        if word == '' or word not in model_w2v.vocab : #or word == '' or word == '' or word == '' or word == '海口' or word == '群眾' or word == '中歸' or word == '這也能' or word == '經不住' or word == '' or word == '海天' or  word == '388' or word == '海興' or word == '彭年' or word == '場在' or word == '給凍醒' or word == '借殼'or word == '1702'or word == '1710'  or word == '剛為' or word == '568' or word == '我主':
                            ary_train[i1][i2][i3] = np.asarray([0.0] * w2v_dim)
                        else :
                            ary_train[i1][i2][i3] = model_w2v[word]
                        i3 += 1

                else :
                    ary_train[i1][i2] = np.zeros((len_sentence,w2v_dim))
                i2 += 1
        else :
            lst_temp = row.split(',')
            i2 = 0
            for sentence in lst_temp :
                lst_temp2 = sentence.split(' ')
                i3 = 0
                if lst_temp2 != [''] :
                    '''
                    if len(lst_temp2) > 50 :
                        print(i1)
                        print(len(lst_temp2))
                        print (row)
                        #print (lst_temp)
                        #print (lst_temp2)
                    '''
                    for word in lst_temp2 :
                        if word == '' or word not in model_w2v.vocab : #or word == '' or word == '' or word == '' or word == '海口' or word == '群眾' or word == '中歸' or word == '這也能' or word == '經不住' or word == '' or word == '海天' or  word == '388' or word == '海興' or word == '彭年' or word == '場在' or word == '給凍醒' or word == '借殼'or word == '1702'or word == '1710'  or word == '剛為' or word == '568' or word == '我主':
                            ary_test[i1-200][i2][i3] = np.asarray([0.0] * w2v_dim)
                        else :
                            ary_test[i1-200][i2][i3] = model_w2v[word]
                        i3 += 1
                else :
                    ary_test[i1-200][i2] = np.zeros((len_sentence,w2v_dim))
                i2 += 1
        i1 += 1

    return (ary_train,ary_test)

def set_new_predictions(th, predictions, Y_train_n=None) : # th means threshold
    s = 0
    index = 0
    ans = []
    if Y_train_n is None :
        for p in predictions :
            if p > th :
                ans.append(1)
            elif p < 1.0-th :
                ans.append(-1)
            else :
                ans.append(0)

            index = index + 1
    else :
        for p in predictions :
            if p > th :
                ans.append(1)
                if Y_train_n[index] == ans[index] :
                    s = s + 1
            elif p < 1-th :
                ans.append(-1)
                if Y_train_n[index] == ans[index] :
                    s = s + 1
            else :
                ans.append(0)
                if Y_train_n[index] == ans[index] :
                    s = s + 1
            index = index + 1

    return (s, ans)

def create_model(X_train,max_comma,len_sentence,w2v_dim) :

    # Embedding dimensions.
    row_hidden = 300
    col_hidden = 300

    d2, d3, d4 = X_train.shape[1:] # 其實也等於上面三個參數
    # 4D input.
    x = Input(shape=(d2,d3,d4))

    #model = Sequential()
    #model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))
    '''
    encoded_rows = TimeDistributed(LSTM(row_hidden,return_sequences=True))(x)
    encoded_rows_2 = TimeDistributed(LSTM(row_hidden,return_sequences=True))(encoded_rows)
    encoded_rows_3 = TimeDistributed(LSTM(row_hidden))(encoded_rows_2)
    encoded_columns = LSTM(col_hidden,return_sequences=True)(encoded_rows_3)
    normal_columns = BatchNormalization()(encoded_columns)
    encoded_columns_2 = LSTM(col_hidden,return_sequences=True)(normal_columns)
    encoded_columns_3 = LSTM(col_hidden)(encoded_columns_2)

    # Final predictions and model.
    prediction = Dense(1, activation='linear')(encoded_columns_3)
    model = Model(x, prediction)
    '''

    '''
    encoded_rows = TimeDistributed(LSTM(row_hidden))(x)
    encoded_columns = LSTM(col_hidden)(encoded_rows)
    prediction = Dense(1, activation='sigmoid')(encoded_columns)
    model = Model(x, prediction)
    '''

    '''
    model = Sequential()
    model.add(TimeDistributed(LSTM(row_hidden, batch_input_shape=(d2,d3,d4))))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(LSTM(col_hidden))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    '''

    encoded_rows = TimeDistributed(LSTM(row_hidden))(x)
    D1 = Dropout(0.2)(encoded_rows)
    nol_encoded_rows = BatchNormalization()(D1)
    encoded_columns = LSTM(col_hidden)(nol_encoded_rows)
    D2 = Dropout(0.2)(encoded_columns)
    normal_columns = BatchNormalization()(D2)
    # Final predictions and model.
    prediction = Dense(1, activation='linear')(normal_columns)
    #nol_prediction = BatchNormalization()(prediction)
    model = Model(x, nol_prediction)


    model.compile(#loss='mean_squared_error',
                  loss='binary_crossentropy',
                  #optimizer='rmsprop',
                  optimizer='adam',
                  metrics=['acc']) #'mae'


    return model




if __name__ == '__main__' :
    print ("start time : " ,datetime.datetime.now())
    start_time = time.time()

    w2v_dim = 300
    max_comma = 20
    epochs = 100
    len_sentence=65
    th = 0.75

    '''
    #preprocessing
    train = pd.read_csv("./data/aspect_review_seg.csv")#,header=None,index_col=False)
    test = pd.read_csv("./data/test_seg.csv")#,header=None,index_col=False)
    train = comma_reset(train,max_comma)
    test = comma_reset(test,max_comma)
    train = aspect_to_vec(train)
    test.to_csv('./data/test_final_2.csv',index=False)
    train.to_csv('./data/train_final_2.csv',index=False)
    w2v(w2v_dim, train, test)
    '''


    question = pd.read_csv("./data/test.csv")
    ans = pd.read_csv("./data/sample_submission.csv")
    train = pd.read_csv("./data/train_final_2.csv")#,header=None,index_col=False)
    test = pd.read_csv("./data/test_final_2.csv")#,header=None,index_col=False)
    '''
    preprocessing
    #X_train,X_test = Review_to_vec(train, test, model_w2v, w2v_dim, max_comma, len_sentence)
    #np.save('./data/X_train',X_train)
    #np.save('./data/X_test',X_test)
    '''
    X_train = np.load('./data/X_train'+'.npy')
    X_test = np.load('./data/X_test'+'.npy')


    #w2v(w2v_dim, train, test)
    model_w2v = word2vec.load('data/word2vec_corpus.bin')
    '''
    for item in ['服務' , '環境' , '價格' , '交通' , '餐廳'] :
        indexes, metrics = model_w2v.cosine(item)
        print(model_w2v.generate_response(indexes, metrics).tolist())
        print('\n')
    '''
    print('XD')
    #0.服務, 1.環境, 2.價格, 3.交通, 4.餐廳
    Y_train = []
    models = []
    predictions = []
    for index in range(5) :
        print(index)
        Y_train.append(Y_select(train['aspect_vec'],index))
        print(str(index) +'-2')
        models.append(create_model(X_train,max_comma,len_sentence,w2v_dim))
        models[index].fit(X_train, Y_train[index], epochs=epochs, verbose=2)
        predictions.append(models[index].predict(X_test))

    (s0, ans0) = set_new_predictions(th,predictions[0],None)
    (s1, ans1) = set_new_predictions(th,predictions[1],None)
    (s2, ans2) = set_new_predictions(th,predictions[2],None)
    (s3, ans3) = set_new_predictions(th,predictions[3],None)
    (s4, ans4) = set_new_predictions(th,predictions[4],None)


    item = ['服務', '環境', '價格', '交通', '餐廳']
    #1.服務, 2.環境, 3.價格, 4.交通, 5.餐廳
    for i in range(len(question)) :
        if question['Aspect'][i] == item[0] :
            ans['Label'][i] = ans0[i]
        elif question['Aspect'][i] == item[1] :
            ans['Label'][i] = ans1[i]
        elif question['Aspect'][i] == item[2] :
            ans['Label'][i] = ans2[i]
        elif question['Aspect'][i] == item[3] :
            ans['Label'][i] = ans3[i]
        elif question['Aspect'][i] == item[4] :
            ans['Label'][i] = ans4[i]

    for k in range(5) :
        for i in range(10) :
            print(predictions[k][i*10])


    ans.to_csv('ans_double_lstm_20comma_128dim_20epochs_th0.75.csv',index=False)




    #w2v(w2v_dim)


    print ("finish time : " , datetime.datetime.now())
    print ("running time is %s seconds" %(time.time()-start_time) )
