"""
Created on Thu Nov 07 20:17:36 2022

@author: @iamrmayur
"""
# Importing the Libraries
import nltk.corpus
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

from spacy.lang.en.tokenizer_exceptions import word
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore') # setting ignore as a parameter

# Importing the Dataset
df = pd.read_csv('Productdetails.csv')
df
df.shape
df.info()
df.describe()

# Visualization some insights from Raw Data
df['Product_Type'].value_counts()
sns.histplot(df['Product_Type'])
plt.show()

df['Sentiment'].value_counts()
sns.histplot(df['Sentiment'])
plt.show()

# Plotting Pie Chart
values = df['Product_Type'].value_counts()     # Counting the unique values frequency
labels = df['Product_Type'].unique().tolist()  # Creating the unique value labels
explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0)     # Exploding the first slice

# Creating the Pie Chart with included exploding slice
plt.pie(values, labels = labels, explode = explode, radius = 1)

# Wordcloud from Raw Data
wordc = ' '.join(df['Product_Description'])

def plot_cloud(wordcloud):
    # Setting figure size
    plt.figure(figsize=(40,30))

    # Display Image
    plt.imshow(wordcloud)

    # No axis details
    plt.axis("off")

# Generating Wordcloud
wordcloud = WordCloud(width = 3000, height=1500, background_color='black', max_words=400, colormap='Set2').generate(wordc)
plot_cloud(wordcloud)

# Top 20 Most Common words from the above wordcloud  i.e., Raw Data
from collections import Counter
# p = Counter(' '.join(df['Product_Description']).split()).most_common(20)
Counter(wordc.split()).most_common(20)
result = pd.DataFrame(Counter(wordc.split()).most_common(20), columns = ['Word','Frequency'])
print(result)

counter = []
for string in df.Product_Description:
    counter.append(string.count(' ') + 1)  # Num of spaces + 1

df['num_words'] = counter  # add the column
df.head(5)

# Cleaning the Dataset
# Text Preprocessing Techniques
import re

def cleantext(text):
    text = re.sub(r"â€™", "", text)             # Remove Mentions
    text = re.sub(r"#", "", text)               # Remove Hashtags Symbol
    text = re.sub(r"\w*\d\w*", "", text)        # Remove numbers
    text = re.sub(r"https?:\/\/\S+", "", text)  # Remove The Hyper Link
    text = re.sub(r"______________", "", text)  # Remove _____
    text=re.sub(r"^a-zA-z0-9","",text)
    text=re.sub(r"[^\w\s]","",text)

    return text

from nltk.tokenize import word_tokenize

df['clean_text'] = df.apply(lambda x: cleantext(x['Product_Description']), axis = 1)
df['clean_text']

# Contractions
import contractions
df['no_contract'] = df['clean_text'].apply(lambda x: [contractions.fix(word) for word in x.split()])
df['no_contract']

# Tokenization
from nltk.tokenize import word_tokenize
df['tokenized']  = df['clean_text'].apply(word_tokenize)

# Lower Case Conversion
df['lower'] = df['tokenized'].apply(lambda x: [word.lower() for word in x])

# Joining df['lower']
df['lower'] = [' '.join(map(str,i)) for i in df['lower']]

df.head(5)

# Stopwords
import nltk
from nltk.corpus import stopwords
from wordcloud import STOPWORDS

stopwords = nltk.corpus.stopwords.words('english')
newstopwords = ["SXSW", "sxsw", "link","iPhone", "upad", "Apple popup" , "RT mention", "RT", "rt", "sxsw sxsw", "Google", "DesignerÛªs" , "link sxsw", "iPad launch", "Social Network", "sxsw apple", "amp","mention google", "via mention", "called circles" , "popup store", "link via", "sxsw sxswi", "downtown austin", "ûïmention" , "sxswi", "marissa mayer", "an iPad", "Circles Possibly", "Austin for","new iPad", "iPad at", "temporary store" , "New UberSocial", "Apple i", "Apple", "popup store", "in Austin", "Called Circles", "Network Called", "Social Network", "Austin","iPad", "Apple Store", "New Social", "sxswÛ", "Facebook", "Circles Possibly", "downtown Austin", "ipad design", "designerûªs", "Marissa Mayer"] + list(stopwords)
list(newstopwords)

stops = r'\b({})\b'.format('|'.join(newstopwords))
df['nostop'] = df['lower'].str.replace(stops, '').str.replace('\s+', ' ')
df.head()

# Now Generating Wordcloud using df['nostop']

wordc = ' '.join(df['nostop'])
wordcloud = WordCloud(width = 3000, height = 1500, background_color = 'black', stopwords = newstopwords, max_words = 400, colormap='Set2').generate(wordc)
plot_cloud(wordcloud)

# Stemming
from nltk.stem import PorterStemmer
st = PorterStemmer()
# stemming every word
df['stemmed'] = df['nostop'].apply(lambda x: ' '.join([st.stem(word) for word in x.split()]))
df['stemmed']

# Lemmatization
from textblob import Word
df['lemma'] = df['stemmed'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

stops = r'\b({})\b'.format('|'.join(newstopwords))
df['lemma'] = df['lemma'].str.replace(stops, '').str.replace('\s+', ' ')
df['lemma'].head(5)

# Top 20 Bi-Grams
from sklearn.feature_extraction.text import CountVectorizer

def top_2_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(2,2),  #for tri-gram, put ngram_range=(3,3)
            max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1],
                reverse=True)
    return words_freq[:n]

from sklearn.feature_extraction.text import CountVectorizer
top2_words = top_2_words(df["lemma"], n=200) #top 200
top2_df = pd.DataFrame(top2_words)
top2_df.columns=["Bi-gram", "Freq"]
top2_df.head(20)

# Plotting the Top 20 Bi-Grams
top2_bigram = top2_df.iloc[0:20,:]
fig = plt.figure(figsize = (20, 5))
plot=sns.barplot(x=top2_bigram["Bi-gram"],y=top2_bigram["Freq"]);
plot.set_xticklabels(rotation=70,labels = top2_bigram["Bi-gram"]);

# Top 20 Tri-Grams
def top_3_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(3,3),
           max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1],
                reverse=True)
    return words_freq[:n]

top3_words = top_3_words(df["lemma"], n=200)
top3_df = pd.DataFrame(top3_words)
top3_df.columns=["Tri-gram", "Freq"]
top3_df.head(20)

# Plotting the Top 20 Tri-Grams
top20_trigram = top3_df.iloc[0:20,:]
fig = plt.figure(figsize = (20, 5))
plot=sns.barplot(x=top20_trigram["Tri-gram"],y=top20_trigram["Freq"]);
plot.set_xticklabels(rotation=90,labels = top20_trigram["Tri-gram"]);

# Top 20 Common Words after cleaning the data
from collections import Counter
p = Counter(" ".join(df['lemma']).split()).most_common(20)
result = pd.DataFrame(p, columns = ['Word','Frequeny'])
print(result)

# Polarity and Subjectivity
from textblob import TextBlob
df['Polarity'] = df['Product_Description'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['Subjectivity'] = df['Product_Description'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# Function to analyze the reviews
def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

df['Analysis'] = df['Polarity'].apply(getAnalysis)

# plotting graph for Polarity (Negative, Neutral, Positive)
Negative_senti = df[df['Polarity']>0]
Neutral_senti = df[df['Polarity']==0]
Positive_senti = df[df['Polarity']<0]

df['Analysis'].value_counts().plot(kind='bar')   # Bar Plot
sns.lmplot (x='Polarity', y='Sentiment', data=df, fit_reg=True)   # Scatter Plot

# Polarity Distribution
plt.figure(figsize=(20,10))
plt.margins(0.04)
plt.xlabel('Polarity', fontsize=15)
plt.xticks(fontsize=20)
plt.ylabel('Frequency', fontsize=15)
plt.yticks(fontsize=20)
plt.hist(df['Polarity'], bins=40)
plt.title('Polarity Distribution', fontsize=20)
plt.show()

# Positive reviews Wordcloud
from wordcloud import WordCloud
wc = WordCloud(width=3000, height=1500, min_font_size=10, max_words=300, stopwords=newstopwords, background_color='black')
Positive = wc.generate(df[df['Polarity']>0]['Product_Description'].str.cat(sep=""))

plt.figure(figsize=(10,10))
plt.imshow(Positive)
plt.title('Positive Reviews')
plt.show()

# Negative Reviews Wordcloud
Negative=wc.generate(df[df['Polarity']<0]['Product_Description'].str.cat(sep=""))

plt.figure(figsize=(10,10))
plt.imshow(Negative)
plt.title('Negative Reviews')
plt.show()

# Neutral Reviews Wordcloud
Neutral = wc.generate(df[df['Polarity']==0]['Product_Description'].str.cat(sep=""))

plt.figure(figsize=(10,10))
plt.imshow(Neutral)
plt.title('Neutral Reviews')
plt.show()

# Pivot Table
df.pivot_table(columns=['Product_Type'], values=['Polarity','Subjectivity'])

df.pivot_table(columns=['Sentiment'], values=['Polarity','Subjectivity'])

temp = df[['num_words','Product_Type','Sentiment','Polarity','Subjectivity']]
temp

# feature Extraction
Negative_senti.head(5)
Neutral_senti.head(5)
Positive_senti.head(5)

"""
# PCA Principal Component Analysis
#scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(temp)
#tranforming the values
scaled_data = scaler.transform(temp)
#implementing PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)
#identifying the number of columns
print("The shape of original dataset:",scaled_data.shape)
print("The shape of dataset after implementing PCA:",x_pca.shape)

x_pca

"""
# words into vector -BOW,Tf-idf,Wordevec
# BOW/Count Vectorization on positive sentiments/reviews

Positive_senti = [lemma.strip() for lemma in Positive_senti.lemma] # remove both the leading and the trailing characters
Positive_senti = [lemma for lemma in Positive_senti if lemma] # removes empty strings, because they are considered in Python as False
Positive_senti[0:10]

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(Positive_senti)

word_freq_df = pd.DataFrame({'term': vectorizer.get_feature_names_out(), 'occurrences':np.asarray(X.sum(axis=0)).ravel().tolist()})
word_freq_df.sort_values('occurrences',ascending=False)

# BOW on Negative words
Negative_senti=Negative_senti['lemma']
Negative_senti=pd.DataFrame(data=Negative_senti)

Negative_senti = [lemma.strip() for lemma in Negative_senti.lemma] # remove both the leading and the trailing characters
Negative_senti = [lemma for lemma in Negative_senti if lemma] # removes empty strings, because they are considered in Python as False
Negative_senti[0:10]

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(Negative_senti)

print(vectorizer.vocabulary_)

word_freq_df = pd.DataFrame({'term': vectorizer.get_feature_names_out(), 'occurrences':np.asarray(X.sum(axis=0)).ravel().tolist()})
word_freq_df.sort_values('occurrences',ascending=False)

# TFidf Vectorizer on Positive Reviews
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer_n_gram_max_features = TfidfVectorizer(norm="l2",analyzer='word', ngram_range=(1,3), max_features = 5000)
tf_idf_matrix_n_gram_max_features =vectorizer_n_gram_max_features.fit_transform(Positive_senti)

print(vectorizer_n_gram_max_features.get_feature_names_out())
print(tf_idf_matrix_n_gram_max_features.toarray())

# TFidf Vectorizer on Negative Reviews
vectorizer_n_gram_max_features = TfidfVectorizer(norm="l2",analyzer='word', ngram_range=(1,3), max_features = 5000)
tf_idf_matrix_n_gram_max_features =vectorizer_n_gram_max_features.fit_transform(Negative_senti)

print(vectorizer_n_gram_max_features.get_feature_names_out())
print(tf_idf_matrix_n_gram_max_features.toarray())

df.head()
temp.head()


corpus = df['lemma'].tolist()
corpus

df['Analysis'] = df['Analysis'].replace({'Negative': -1})
df['Analysis'] = df['Analysis'].replace({'Positive': 1})
df['Analysis'] = df['Analysis'].replace({'Neutral': 0})

senti_into_number_form=df['Analysis']

senti_into_number_form.head(50)

# Corpus converted into array using TFidf
vectorizer4 = TfidfVectorizer(max_features=8000)
idf = vectorizer4.fit_transform(corpus).toarray()
idf

import pickle
pickle_out=open('vectorizer4.pkl','wb')
pickle.dump(vectorizer4,pickle_out)
pickle_out.close()

xtfidf = pd.DataFrame(idf)
xtfidf

# Target var
ytfidf=df['Sentiment']
ytfidf.value_counts() # checking whether data is balanced or imbalanced

# Test and Split Training Data
from sklearn.model_selection import train_test_split

x_traintfidf, x_testtfidf,y_traintfidf,y_testtfidf = train_test_split(xtfidf,ytfidf, test_size=0.33,random_state=0)
x_traintfidf.shape,y_traintfidf.shape, x_testtfidf.shape,y_testtfidf.shape

# Balancing the splitted (tfidf) data using SMOTE method
from imblearn.over_sampling import SMOTE

upsample = SMOTE()
x_traintfidf1, y_traintfidf1 = upsample.fit_resample(x_traintfidf, y_traintfidf)

#target y
ytfidf1=df['Sentiment']
ytfidf1.value_counts()

# Model building with balanced data using Tfidf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score,accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
1. Logistic Regression
"""
from sklearn.linear_model import LogisticRegression

lr= LogisticRegression()
lr.fit(x_traintfidf1,y_traintfidf1)
y_pred_test_lr=lr.predict(x_testtfidf)
y_pred_train_lr=lr.predict(x_traintfidf1)

# accuracy score
accuracy_train_LR=accuracy_score(y_traintfidf1,y_pred_train_lr)*100
accuracy_test_LR= accuracy_score(y_testtfidf, y_pred_test_lr) * 100
print('Accuracy of Training data =',accuracy_train_LR)
print("Accuracy of Test data =", accuracy_test_LR)

print(classification_report(y_testtfidf, y_pred_test_lr))

from sklearn import metrics
classes = np.unique(y_testtfidf)
cm0 = metrics.confusion_matrix(y_testtfidf, y_pred_test_lr)
fig, ax = plt.subplots()
sns.heatmap(cm0, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
            cbar=False)
ax.set(xlabel="Pred", ylabel="True", xticklabels=classes,
       yticklabels=classes, title="Confusion matrix")
plt.yticks(rotation=0);

"""
2. Random Forest
"""
from sklearn.ensemble import RandomForestClassifier
RF= RandomForestClassifier()
RF.fit(x_traintfidf1,y_traintfidf1)
y_pred_test_RF=RF.predict(x_testtfidf)
y_pred_train_RF=RF.predict(x_traintfidf1)

# accuracy score
accuracy_train_RF=accuracy_score(y_traintfidf1,y_pred_train_RF)*100
accuracy_test_RF= accuracy_score(y_testtfidf, y_pred_test_RF) * 100
print('Accuracy of Training data =',accuracy_train_RF)
print("Accuracy of Test data =", accuracy_test_RF)

print(classification_report(y_testtfidf, y_pred_test_RF))

from sklearn import metrics
classes = np.unique(y_testtfidf)
cm0 = metrics.confusion_matrix(y_testtfidf, y_pred_test_RF)
fig, ax = plt.subplots()
sns.heatmap(cm0, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
            cbar=False)
ax.set(xlabel="Pred", ylabel="True", xticklabels=classes,
       yticklabels=classes, title="Confusion matrix")
plt.yticks(rotation=0);

"""
3. SVM
"""
from sklearn.svm import LinearSVC
SVM= LinearSVC()
SVM.fit(x_traintfidf1,y_traintfidf1)
y_pred_test_SVM=SVM.predict(x_testtfidf)
y_pred_train_SVM=SVM.predict(x_traintfidf1)

print(classification_report(y_testtfidf, y_pred_test_SVM))

accuracy_train_SVM=accuracy_score(y_traintfidf1,y_pred_train_SVM)*100
accuracy_test_SVM= accuracy_score(y_testtfidf, y_pred_test_SVM) * 100
print('Accuracy of Training data =',accuracy_train_SVM)
print("Accuracy of Test data =", accuracy_test_SVM)

from sklearn import metrics
classes = np.unique(y_testtfidf)
cm0 = metrics.confusion_matrix(y_testtfidf, y_pred_test_SVM)
fig, ax = plt.subplots()
sns.heatmap(cm0, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
            cbar=False)
ax.set(xlabel="Pred", ylabel="True", xticklabels=classes,
       yticklabels=classes, title="Confusion matrix")
plt.yticks(rotation=0);

"""
4. Naive Bayes classifier for multinomial models
"""
from sklearn.naive_bayes import MultinomialNB
MULT_NB= MultinomialNB()
MULT_NB.fit(x_traintfidf1,y_traintfidf1)
y_pred_test_mult_nb=MULT_NB.predict(x_testtfidf)
y_pred_train_mult_nb=MULT_NB.predict(x_traintfidf1)

accuracy_train_MULT_NB=accuracy_score(y_traintfidf1,y_pred_train_mult_nb)*100
accuracy_test_MULT_NB= accuracy_score(y_testtfidf, y_pred_test_mult_nb) * 100
print('Accuracy of Training data =',accuracy_train_MULT_NB)
print("Accuracy of Test data =", accuracy_test_MULT_NB)

print(classification_report(y_testtfidf, y_pred_test_mult_nb))

from sklearn import metrics
classes = np.unique(y_testtfidf)
cm0 = metrics.confusion_matrix(y_testtfidf, y_pred_test_mult_nb)
fig, ax = plt.subplots()
sns.heatmap(cm0, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
            cbar=False)
ax.set(xlabel="Pred", ylabel="True", xticklabels=classes,
       yticklabels=classes, title="Confusion matrix")
plt.yticks(rotation=0);

"""
5. AdaBoost classifier on tfidf features
"""
from sklearn.ensemble import AdaBoostClassifier
ADA= AdaBoostClassifier()
ADA.fit(x_traintfidf1,y_traintfidf1)
y_pred_test_ada=ADA.predict(x_testtfidf)
y_pred_train_ada=ADA.predict(x_traintfidf1)

accuracy_train_ADA=accuracy_score(y_traintfidf1,y_pred_train_ada)*100
accuracy_test_ADA= accuracy_score(y_testtfidf, y_pred_test_ada) * 100
print('Accuracy of Training data =',accuracy_train_ADA)
print("Accuracy of Test data =", accuracy_test_ADA)

print(classification_report(y_testtfidf, y_pred_test_ada))

from sklearn import metrics
classes = np.unique(y_testtfidf)
cm0 = metrics.confusion_matrix(y_testtfidf, y_pred_test_ada)
fig, ax = plt.subplots()
sns.heatmap(cm0, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
            cbar=False)
ax.set(xlabel="Pred", ylabel="True", xticklabels=classes,
       yticklabels=classes, title="Confusion matrix")
plt.yticks(rotation=0);

"""
6. KNN
"""
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

KNN= AdaBoostClassifier()
KNN.fit(x_traintfidf1,y_traintfidf1)
y_pred_test_KNN=KNN.predict(x_testtfidf)
y_pred_train_KNN=KNN.predict(x_traintfidf1)

accuracy_train_KNN=accuracy_score(y_traintfidf1,y_pred_train_ada)*100
accuracy_test_KNN= accuracy_score(y_testtfidf, y_pred_test_ada) * 100
print('Accuracy of Training data =',accuracy_train_KNN)
print("Accuracy of Test data =", accuracy_test_KNN)

print(classification_report(y_testtfidf, y_pred_test_KNN))

from sklearn import metrics
classes = np.unique(y_testtfidf)
cm0 = metrics.confusion_matrix(y_testtfidf, y_pred_test_KNN)
fig, ax = plt.subplots()
sns.heatmap(cm0, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
            cbar=False)
ax.set(xlabel="Pred", ylabel="True", xticklabels=classes,
       yticklabels=classes, title="Confusion matrix")
plt.yticks(rotation=0);

"""
7. XG-Boost
"""

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

XGB = XGBClassifier()
XGB.fit(x_traintfidf1,y_traintfidf1)

# make predictions for test data
y_pred_test_XGB = XGB.predict(x_testtfidf)
y_pred_train_XGB = XGB.predict(x_traintfidf1)

accuracy_train_XGB=accuracy_score(y_traintfidf1,y_pred_train_ada)*100
accuracy_test_XGB= accuracy_score(y_testtfidf, y_pred_test_ada) * 100
print('Accuracy of Training data =',accuracy_train_XGB)
print("Accuracy of Test data =", accuracy_test_XGB)

print(classification_report(y_testtfidf, y_pred_test_XGB))

from sklearn import metrics
classes = np.unique(y_testtfidf)
cm0 = metrics.confusion_matrix(y_testtfidf, y_pred_test_XGB)
fig, ax = plt.subplots()
sns.heatmap(cm0, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
            cbar=False)
ax.set(xlabel="Pred", ylabel="True", xticklabels=classes,
       yticklabels=classes, title="Confusion matrix")
plt.yticks(rotation=0);

"""
Saving the Best Model
"""
filename = 'saved_lr_model.pkl'
pickle.dump(lr,open(filename,'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
loaded_model.predict(x_testtfidf)
