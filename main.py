#Using the text.txt dataset


#Initialization
classes = []
texts = []
converted_classes = []
converted_texts = []

#List of emotions
emotion = [ 'joy', 'fear', 'anger', 'sadness', 'disgust', 'shame', 'guilt' ]

#Converting text into feature vectors

import re
import collections

def feature_conversion (line, _len):
    features = []
    def gen_tokens (n, _line):
      tokens = []
      _len = len(_line)
      for i in range(n, _len): tokens.append(' '.join(_line[i-n:i]))
      return tokens
    line = re.sub('[^0-9a-z#!.?]', ' ', line.lower())
    for n in range(_len[0], _len[1]+1): features += gen_tokens(n, line.split())

    return collections.Counter(features)


# Load entries
def load_entries():
    file = open('text.txt', 'r')
    lines = file.readlines()
    lines = [line.strip() for line in lines]

    for line in lines:
        demarcation = line.find("]")  # separates class and text
        encoding = line[1:demarcation].strip()
        text = line[demarcation + 1:].strip()

        classes.append(' '.join(encoding.split()))
        texts.append(text)


load_entries()
print(classes)
print(texts)

#Converting training samples

def modify ():
  _len = len(texts)
  for i in range (0, _len):
    converted_texts.append(feature_conversion(texts[i], _len=(1, 4)))
    list_emotions = classes[i].split(" ")
    list_emotions = list(map(float, list_emotions))
    for emot in range(0, 7):
      if list_emotions[emot] == 1:
        converted_classes.append(emotion[emot])
        break

modify()

#Split dataset into training and testing sets

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(converted_texts, converted_classes,
                                                    test_size = 0.2, random_state = 42)

from sklearn.feature_extraction import DictVectorizer

dict_vectorizer = DictVectorizer(sparse = True)
x_train = dict_vectorizer.fit_transform(x_train)
x_test = dict_vectorizer.transform(x_test)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

import time

start_time = time.time()

#Configure classical classifiers

def setup_classifiers():
    svc_classifier = SVC(probability=True)
    randfor_classifier = RandomForestClassifier(random_state=42)
    dtree_classifier = DecisionTreeClassifier(random_state=42)
    print("Classifier     Accuracy (Training)   Accuracy(Testing)")

    svc_classifier.fit(x_train, y_train)
    print("SVC", "              ", "{:1.5f}".format(accuracy_score(y_train, svc_classifier.predict(x_train))),
          "              ", "{:1.5f}".format(accuracy_score(y_test, svc_classifier.predict(x_test))))
    dtree_classifier.fit(x_train, y_train)
    print("DecisionTree", "     ", "{:1.5f}".format(accuracy_score(y_train, dtree_classifier.predict(x_train))),
          "             ", "{:1.5f}".format(accuracy_score(y_test, dtree_classifier.predict(x_test))))
    randfor_classifier.fit(x_train, y_train)
    print("RandomForest", "     ", "{:1.5f}".format(accuracy_score(y_train, randfor_classifier.predict(x_train))),
          "             ", "{:1.5f}".format(accuracy_score(y_test, randfor_classifier.predict(x_test))))

    return svc_classifier, randfor_classifier, dtree_classifier


svc_classifier, randfor_classifier, dtree_classifier = setup_classifiers()

print("%s s" % (time.time() - start_time))

svc_res = svc_classifier.predict(x_test)
dtree_res = dtree_classifier.predict(x_test)
randfor_res = randfor_classifier.predict(x_test)

print(svc_res)
print(dtree_res)
print(randfor_res)

#Display confusion matrix

from sklearn.metrics import confusion_matrix

svc_cm = confusion_matrix(y_test, svc_res, labels=emotion)
dtree_cm = confusion_matrix(y_test, dtree_res, labels=emotion)
randfor_cm = confusion_matrix(y_test, randfor_res, labels=emotion)

print("Confusion matrix for SVC:")
print(svc_cm)

print("\nConfusion matrix for DecisionTree:")
print(dtree_cm)

print("\nConfusion matrix for RandomForest:")
print(randfor_cm)

#Finding precision and recall

import numpy as np

recall = np.diag(svc_cm) / np.sum(svc_cm, axis = 1)
precision = np.diag(svc_cm) / np.sum(svc_cm, axis = 0)

print("SVC Recall: ", recall)
print("SVC Precision: ", precision)

recall = np.diag(dtree_cm) / np.sum(dtree_cm, axis = 1)
precision = np.diag(dtree_cm) / np.sum(dtree_cm, axis = 0)

print("\nDecisionTree Recall: ", recall)
print("DecisionTree Precision: ", precision)

recall = np.diag(randfor_cm) / np.sum(randfor_cm, axis = 1)
precision = np.diag(randfor_cm) / np.sum(randfor_cm, axis = 0)

print("\nRandomForest Recall: ", recall)
print("RandomForest Precision: ", precision)


# Sample testing
# 'joy', 'fear', 'anger', 'sadness', 'disgust', 'shame', 'guilt'

def sample_testing():
    samples = ["I have a strange fear of peacocks",
               "He should be ashamed of himself",
               "I forgot about the letter",
               "I am happy for you!",
               "Her pet died last night",
               "She was angry",
               "Someone spread rumours about me",
               "Her feet were on the seat"
               ]
    print("Sample Testing scores: ")

    def predict(line, classifier, classfname):
        print("[", classfname, "] : ",
              classifier.predict(dict_vectorizer.transform(feature_conversion(line, _len=(1, 4))))[0])

    for sample in samples:
        print(sample)
        predict(sample, svc_classifier, "SVC")
        predict(sample, dtree_classifier, "DecisionTree")
        predict(sample, randfor_classifier, "RandomForest")
        print("\n")


sample_testing()

#Modern (NN) techniques

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
import nltk # Natural Language toolkit
from nltk.corpus import stopwords

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem import PorterStemmer
import re

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Embedding,Bidirectional,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

testData =  pd.read_csv('test.txt',sep=";",names=['Comment','Emotion'],encoding="utf-8")
trainData = pd.read_csv('train.txt',sep=";",names=['Comment','Emotion'],encoding = "utf-8")
valData = pd.read_csv('val.txt',sep=";",names=['Comment','Emotion'],encoding = "utf-8")
print("Shape of Train Data: ",trainData.shape)
print("Shape of Val Data: ",valData.shape)
print("Shape of Test Data: ",testData.shape)
trainData.head(5)

# To cehck if the length of the sentences is correlated to emotion or not
trainData['length']=[len(x) for x in trainData['Comment']]
trainData.head(5)

# Plotting the kernel density estimate(kde) plot for the three data sets
all_data={'Train Data':trainData, 'Validation Data':valData,'Test Data':testData}
fig, ax = plt.subplots(1,3, figsize=(30,10))
for i, df in enumerate(all_data.values()):
    df2 = df.copy()
    df2['length'] = [len(x) for x in df2['Comment']]
    sns.kdeplot(data=df2,x='length',hue='Emotion', ax=ax[i])
plt.show()

# Word Cloud is a data visualization technique used for representing text data in which
# the size of each word indicates its frequency or importance.
def words_cloud(wordcloud,df):
    plt.figure(figsize=(10,10))
    plt.title(df+' Word Cloud', size =20)
    plt.imshow(wordcloud)

emotion_list=trainData['Emotion'].unique()
emotion_list

# Creating and displaying the word cloud.
# Here 6 word clouds are made, one for each emotion.
# All the sentences of the same emotion are concatenated using 'join' function, and then the wordcloud is generated.

for emotion in emotion_list:
    text=' '.join([sentence for sentence in trainData.loc[trainData['Emotion'] == emotion,'Comment']])
    wordcloud = WordCloud(width=600,height=600).generate(text)
    words_cloud(wordcloud, emotion)


#Data Pre-processing


# Here the target values are encoded from integer 0 to n-1 classes, so that they can be machine readable.
# For Training Dataset
lb = LabelEncoder()
trainData['Emotion']=lb.fit_transform(trainData['Emotion'])
# For Test Dataset
testData['Emotion']=lb.fit_transform(testData['Emotion'])
# For Validation Dataset
valData['Emotion']=lb.fit_transform(valData['Emotion'])

nltk.download('stopwords')

print(stopwords.words('english'))
stop_words=set(stopwords.words('english'))

max_len=trainData['length'].max()
max_len


# We are calling the Tokenizer library to divide the string into list of substrings.
# The reason why we stem is to shorten the lookup, and normalize sentences.
def text_cleaning(df, column):
    print(df.shape)
    stemmer = PorterStemmer()
    corpus = []

    for text in df[column]:
        text = re.sub("[^a-zA-Z]", " ", text)
        text = text.lower()
        text = text.split()
        text = [stemmer.stem(word) for word in text if word not in stop_words]
        text = " ".join(text)
        corpus.append(text)
        print(corpus)
    one_hot_word = [one_hot(input_text=word, n=11000) for word in corpus]
    pad = pad_sequences(sequences=one_hot_word, maxlen=max_len, padding='pre')
    print(pad.shape)
    return pad

x_train =  text_cleaning(trainData, "Comment")
x_test = text_cleaning(testData, "Comment")
x_val = text_cleaning(valData, "Comment")

x_train[1]

y_train = trainData['Emotion']
y_test = testData['Emotion']
y_val = valData['Emotion']

y_trainc=to_categorical(y_train)
y_testc = to_categorical(y_test)
y_valc = to_categorical(y_val)

#Model Building

model = Sequential()
model.add(Embedding(input_dim = 11000,output_dim=150,input_length=300))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(64,activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(6,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
callback=EarlyStopping(monitor="val_loss",patience=2,restore_best_weights=True)
hist=model.fit(x_train,y_trainc,epochs=10,batch_size=64,validation_data=(x_val,y_valc),verbose=1,callbacks=[callback])

model.evaluate(x_val,y_valc,verbose=1)
model.evaluate(x_test,y_testc,verbose=1)

accuracy=hist.history['accuracy']
val_acc = hist.history['val_accuracy']
loss=hist.history['loss']
val_loss = hist.history['val_loss']
epochs= range(len(accuracy))

plt.plot(epochs,accuracy,'b',label='Training Accuracy')
plt.plot(epochs,val_acc,'r',label='validation Accuracy')
plt.title("Training and Vaidation Accuracy graphs")
plt.legend()
plt.figure()

plt.plot(epochs,loss,'b', label='Training loss')
plt.plot(epochs,val_loss,'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

def sentence_cleaning(sentence):
    stemmer=PorterStemmer()
    corpus = []
    text = re.sub("[^a-zA-Z]", " ",sentence)
    text =  text.lower()
    text = text.split()
    text = [ stemmer.stem(word) for word in text if word not in stop_words]
    text = " ".join(text)
    corpus.append(text)
    one_hot_word=[ one_hot(input_text=word, n = 11000) for word in corpus]
    pad = pad_sequences(sequences=one_hot_word,maxlen=max_len,padding='pre')
    return pad

sentences = [
            "He was speechles when he found out he was accepted to this new job",
            "This is outrageous, how can you talk like that?",
            "I feel like im all alone in this world",
            "He is really sweet and caring"
            ]

for sentence in sentences:
    print(sentence)
    sentence = sentence_cleaning(sentence)
    result = lb.inverse_transform(np.argmax(model.predict(sentence), axis = -1))[0]
    proba = np.max(model.predict(sentence))
    print(f"{result} : {proba}\n\n")

samples = ["I have a strange fear of peacocks",
             "He should be ashamed of himself",
             "I forgot about the letter",
             "I am happy for you!",
              "Her pet died last night",
             "She was angry",
             "Someone spread rumours about me",
             "Her feet were on the seat"
             ]
for sentence in samples:
    print(sentence)
    sentence = sentence_cleaning(sentence)
    result = lb.inverse_transform(np.argmax(model.predict(sentence), axis = -1))[0]
    proba = np.max(model.predict(sentence))
    print(f"{result} : {proba}\n\n")
