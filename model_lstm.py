#encoding=utf-8
import os
import time,datetime
import pandas as pd
import numpy as np
import keras
from keras.layers import LSTM, Embedding, Input, Dense, TimeDistributed
import word2vec

import sys
sys.path.append("/Users/xogo/Desktop/NTU/2017 spring/NLP/project1/NLP_opinion_mining/")
sys.path.append("/Users/xogo/Desktop/NTU/2017 spring/NLP/project1/NLP_opinion_mining/data/")


# Training parameters.
batch_size = 32
num_classes = 10
epochs = 5

# Embedding dimensions.
row_hidden = 128
col_hidden = 128

#keras.utils.np_utils.to_categorical()
#keras.utils.to_categorical



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
        '''
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('。',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('，',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('！',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('!',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('？',',')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('?',',')
        #ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('、',',')
        #ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('\'','')
        '''
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('~',' ')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('～',' ')
        ary.loc[row,'Review'] = ary.loc[row,'Review'].replace('   ',',')
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
                if p2 == -1 or p1 == -1:
                    p2 = 0
                    continue
                ary.loc[row,'Review'] = ary.loc[row,'Review'][:p2] +' '+ ary.loc[row,'Review'][p2+1:]
                t2 = t2 - 1
                if t2 <= num_max - 1 :
                    break
            #t2 = int(t2/2) + 1
    return ary
def w2v(w2v_dim=300) :
    aspect_review = pd.read_csv('./data/train_final_1.csv')
    polarity_review = pd.read_csv('data/polarity_review_seg.csv')
    test_review = pd.read_csv('./data/test_final_1.csv')
    test_review = test_review.groupby('Review_id').first()
    reviews = aspect_review['Review'].append(polarity_review['Review']).append(test_review['Review'])
    print('#reviews:', len(reviews))

    with open("data/word2vec_corpus.tmp", "w") as f:
            f.write(("\n".join(reviews)+"\n"))
    print('running word2vec ...')
    #word2vec.word2phrase('data/word2vec_corpus.tmp', 'data/word2vec_corpus_phrases', verbose=True)
    word2vec.word2vec('data/word2vec_corpus.tmp', 'data/word2vec_corpus.bin', size=w2v_dim, verbose=True, window=6, cbow=1, binary=1, min_count=1, iter_=10)

def Review_to_vec(train,test,max_comma,w2v_dim,model_w2v) :
    review_train = np.array(train.loc[:,'Review'])
    aspect_vec_train = np.array(train.loc[:,'aspect_vec'])
    review_test = np.array(test.loc[:,'Review'])
    review = np.append(review_train,review_test,axis=0)

    lst_train = []
    lst_test = []
    index = 0
    c = 0
    for row in review :
        lst_temp = row.split(',')
        s_vec = []
        for sentence in lst_temp :
            lst_temp2 = sentence.split(' ')
            w_vec = []
            if lst_temp2 != [''] :
                for word in lst_temp2 :
                    if word == '' : #or word == '' or word == '' or word == '' or word == '海口' or word == '群眾' or word == '中歸' or word == '這也能' or word == '經不住' or word == '' or word == '海天' or  word == '388' or word == '海興' or word == '彭年' or word == '場在' or word == '給凍醒' or word == '借殼'or word == '1702'or word == '1710'  or word == '剛為' or word == '568' or word == '我主':
                        continue
                    elif word not in model_w2v.vocab :
                        w_vec.append([[0.0] * w2v_dim])
                        c = c + 1
                        print(c)
                        print(word)
                    else :
                        #print(model_w2v[word].tolist())
                        w_vec.append(model_w2v[word].tolist())
            else :
                #print('...')
                w_vec = [[0.0] * w2v_dim]
            s_vec.append(w_vec)
        if index < 200 :
            lst_train.append(s_vec)
        else :
            lst_test.append(s_vec)
        #lst_CountVec = lst_CountVec + lst_temp
        index = index + 1

    vec_train = vec_ary[:(200-num_train_omc)*max_comma][:].toarray()
    vec_test = vec_ary[(200-num_train_omc)*max_comma:][:].toarray()

    X_train = np.reshape(vec_train, (int(vec_train.shape[0]/max_comma),max_comma,vec_train.shape[1]))
    X_test = np.reshape(vec_test, (int(vec_test.shape[0]/max_comma),max_comma,vec_test.shape[1]))

    Y_train = lst_avt

    return (X_train,Y_train,X_test)



if __name__ == '__main__' :
    print ("start time : " ,datetime.datetime.now())
    start_time = time.time()

    w2v_dim = 300
    max_comma = 20
    epochs = 20

    '''
    #preprocessing
    train = pd.read_csv("./data/aspect_review_seg.csv")#,header=None,index_col=False)
    test = pd.read_csv("./data/test_seg.csv")#,header=None,index_col=False)
    question = pd.read_csv("./data/test.csv")
    ans = pd.read_csv("./data/sample_submission.csv")
    train = comma_reset(train,max_comma)
    test = comma_reset(test,max_comma)
    train = aspect_to_vec(train)
    test.to_csv('./data/test_final_1.csv',index=False)
    train.to_csv('./data/train_final_1.csv',index=False)
    w2v()
    '''

    train = pd.read_csv("./data/train_final_1.csv")#,header=None,index_col=False)
    test = pd.read_csv("./data/test_final_1.csv")#,header=None,index_col=False)
    model_w2v = word2vec.load('data/word2vec_corpus.bin')
    Review_to_vec(train,test,max_comma,w2v_dim,model_w2v)

    #w2v(w2v_dim)
