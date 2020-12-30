---
title: "Text Classification (NLTK)"
date: 2019-12-10
tags: [nltk, text classification, nlp]
header:
  image: "/images/nlp.png"
excerpt: "NLTK, Text Classification, Data Science, NLP"
mathjax: "true"
---
## OBJECTIVE: Classification/Prediction of texts which belongs to nine different categories/books of the Gutenberg’s digital corpus by using various ML classification algorithm and then choosing the best algorithm for the purpose depending on their performance.

The main steps involved are : 

1. DATA PREPARATION & CLEANING
2. WORD TO VECTOR TRANFORMATION
3. MODELING
4. PERFORMANCE EVALUATION
5. ERROR ANALYSIS

A more detailed explation of this project can be found in the repository [here](https://github.com/gaurav-arena/Text-Classification-NLTK)

**IMPORTING THE NECESSARY LIBRARIES**


```python
import nltk
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn import naive_bayes
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
```

**DOWNLOADING THE RELEVANT RESOURCES FROM THE NLTK CORPUS**


```python
nltk.download('gutenberg')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

**CHECKING THE CONTENT OF THE GUTENBERG CORPUS**


```python
nltk.corpus.gutenberg.fileids()
```




    ['austen-emma.txt',
     'austen-persuasion.txt',
     'austen-sense.txt',
     'bible-kjv.txt',
     'blake-poems.txt',
     'bryant-stories.txt',
     'burgess-busterbrown.txt',
     'carroll-alice.txt',
     'chesterton-ball.txt',
     'chesterton-brown.txt',
     'chesterton-thursday.txt',
     'edgeworth-parents.txt',
     'melville-moby_dick.txt',
     'milton-paradise.txt',
     'shakespeare-caesar.txt',
     'shakespeare-hamlet.txt',
     'shakespeare-macbeth.txt',
     'whitman-leaves.txt']

**DATA PREPARATION & CLEANING:**

The data needs to be prepared and cleaned and it is an essential part of this assignment. Nine different books are selected based on their authors from the Gutenberg's digital corpus for having nine distinct classes with the aim of having distinct targets for the model. The NLTK library includes a small selection of texts from the Project Gutenberg electronic text archive, which contains some 25,000 free digital books. Thus to begin with, first the NLTK library was imported and nine distinct books/texts were selected from the Gutenberg corpus based on their authors.

**ASSIGNING THE SELECTED TEXTS TO VARIABLES**


```python
text1 =  nltk.corpus.gutenberg.raw('austen-emma.txt')
text2 = nltk.corpus.gutenberg.raw('bible-kjv.txt')
text3 = nltk.corpus.gutenberg.raw('whitman-leaves.txt')
text4 = nltk.corpus.gutenberg.raw('milton-paradise.txt')
text5 = nltk.corpus.gutenberg.raw('melville-moby_dick.txt')
text6 = nltk.corpus.gutenberg.raw('edgeworth-parents.txt')
text7 = nltk.corpus.gutenberg.raw('chesterton-thursday.txt')
text8 = nltk.corpus.gutenberg.raw('shakespeare-hamlet.txt')
text9 = nltk.corpus.gutenberg.raw('bryant-stories.txt')
```
The selected texts are then sentence tokenized and then a function was defined, named as ‘word_list’ and it has the capacity of removing unwanted punctuations and stop-words (as these would corrupt the training model and are therefore needed to be removed). The sentences were further tokenized to words and lemmatizing took place which is the process by which the all the words are reduced to the root words by using the morphological analysis of words and the vocabulary depending upon whether it’s a noun verb or an adjective. It helps in reducing the size of the data and also makes it convenient to keep track the specific words instead the same words used in different forms. Since there is a requirement for only words there’s a need to remove all data that doesn’t count as words e.g. numbers, punctuations etc. It reduces the size of the data as well as it makes the process faster. All the words are also transformed to lower case so that the computer doesn’t recognise same words in different cases as unlike.

**VIEWING THE NUMBER OF WORDS IN EACH TEXT**


```python
number_words_texts=[(len(nltk.word_tokenize(text1)),'austen-emma'),(len(nltk.word_tokenize(text2)),'bible-kjv'), (len(nltk.word_tokenize(text3)),'whitman-leaves'), 
                    (len(nltk.word_tokenize(text4)),'milton-paradise'), (len(nltk.word_tokenize(text5)),'melville-moby_dick'), 
                    (len(nltk.word_tokenize(text6)),'edgeworth-parents'), (len(nltk.word_tokenize(text7)),'chesterton-thursday'), 
                    (len(nltk.word_tokenize(text8)),'shakespeare-hamlet'), (len(nltk.word_tokenize(text9)),'bryant-stories')]
print(number_words_texts)
```

    [(191673, 'austen-emma'), (946812, 'bible-kjv'), (149198, 'whitman-leaves'), (95709, 'milton-paradise'), (254989, 'melville-moby_dick'), (209090, 'edgeworth-parents'), (69408, 'chesterton-thursday'), (36326, 'shakespeare-hamlet'), (55621, 'bryant-stories')]
    


```python
n_o_w = pd.DataFrame(number_words_texts)
n_o_w = n_o_w.rename(columns={0: "Number of words", 1: "Text"})
print(n_o_w)
```

       Number of words                 Text
    0           191673          austen-emma
    1           946812            bible-kjv
    2           149198       whitman-leaves
    3            95709      milton-paradise
    4           254989   melville-moby_dick
    5           209090    edgeworth-parents
    6            69408  chesterton-thursday
    7            36326   shakespeare-hamlet
    8            55621       bryant-stories
    


```python
sns.barplot(n_o_w['Text'],n_o_w['Number of words'],label="Number of Words")
plt.xticks(rotation=90)
plt.show()
```


![png](/images/Text_Classification_NLTK_files/Text_Classification_NLTK_12_0.png)


**TOKENIZING THE SELECTED TEXTS INTO SENTENCES**


```python
sent1 = nltk.sent_tokenize(text1)
sent2 = nltk.sent_tokenize(text2)
sent3 = nltk.sent_tokenize(text3)
sent4 = nltk.sent_tokenize(text4)
sent5 = nltk.sent_tokenize(text5)
sent6 = nltk.sent_tokenize(text6)
sent7 = nltk.sent_tokenize(text7)
sent8 = nltk.sent_tokenize(text8)
sent9 = nltk.sent_tokenize(text9)
```


**PREPROCESSING OF THE DATA :**

**DEFINING A FUNCTION (word_list) TO REMOVE STOPWORDS, NUMBERS, PUNCTUATIONS AS WELL AS LEMMATIZE**


```python
wordlemmatize = WordNetLemmatizer()

def word_list(sent):
    sentn = ''
    sentn = sentn.join(sent)
    sentn=sentn.replace('.',' ').replace(',',' ').replace('!',' ').replace('?',' ').replace('--',' ').replace('-',' ').replace(';',' ').replace("'",' ').replace('"',' ').replace("_",' ').replace(':',' ').replace('(',' ').replace(')',' ').replace('0',' ').replace('1',' ').replace('2',' ').replace('3',' ').replace('4',' ').replace('5',' ').replace('6',' ').replace('7',' ').replace('8',' ').replace('9',' ')#REPLACING OF UNWANTED CHARACTERS
    words = nltk.word_tokenize(sentn) #TOKENIZING THE SENTENCES INTO WORDS
    stop_words = set(stopwords.words('english'))

    wordlen = []
    words = [w.lower() for w in words if not w in stop_words] #REMOVAL OF STOPWORDS FROM THE TEXT 
    for word in words:
        wordsv = wordlemmatize.lemmatize(word, pos='v') #LEMMATIZING OF THE TEXT BASED ON VERBS
        wordsa = wordlemmatize.lemmatize(wordsv, pos='a') #LEMMATIZING OF THE TEXT BASED ON ADJECTIVES
        words = wordlemmatize.lemmatize(wordsa, pos='n') #LEMMATIZING OF THE TEXT BASED ON NOUNS
        wordlen.append(words) #APPENDING THE WORDS AFTER THE REMOVAL OF STOPWORDS, UNWANTED PUNTUATIONS, SYMBOLS, NUMBERS AND LEMMANTIZING 
    return wordlen

```

**APPLYING THE FUNCTION (word_list) TO THE TOKENIZED SENTENCES**



```python
words1 = word_list(sent1)
words2 = word_list(sent2)
words3 = word_list(sent3)
words4 = word_list(sent4)
words5 = word_list(sent5)
words6 = word_list(sent6)
words7 = word_list(sent7)
words8 = word_list(sent8)
words9 = word_list(sent9)
```

**CHECKING FOR THE LENGHT OF THE TEXTS AFTER THE FIRST FUNTION WAS APPLIED**


```python
number_words_texts=[(len(words1),'austen-emma'),(len(words2),'bible-kjv'),(len(words3),'whitman-leaves'), (len(words4), 'milton-paradise'),
                    (len(words5),'melville-moby_dick'), (len(words6), 'edgeworth-parents'), (len(words7),'chesterton-thursday'),
                    (len(words8),'shakespeare-hamlet'),(len(words9),'bryant-stories')]
n_o_w = pd.DataFrame(number_words_texts)
n_o_w = n_o_w.rename(columns={0: "Number of words", 1: "Text"})
print(n_o_w)
```

       Number of words                 Text
    0            81493          austen-emma
    1           413693            bible-kjv
    2            75686       whitman-leaves
    3            52974      milton-paradise
    4           118804   melville-moby_dick
    5            87234    edgeworth-parents
    6            31614  chesterton-thursday
    7            18793   shakespeare-hamlet
    8            24538       bryant-stories
    

**VISUALIZATION OF THE LENGHT OF THE TEXTS**


```python
sns.barplot(n_o_w['Text'],n_o_w['Number of words'],label="Number of Words")
plt.xticks(rotation=90)
plt.show()
```


![png](/images/Text_Classification_NLTK_files/Text_Classification_NLTK_23_0.png)

As the number of words in each of these 9 texts/books are not same, another function is then defined which is responsible for breaking down a book/text into multiple documents of 100 words each, and then selecting 150 random documents for each book/text. This function is used to make sure that the trained model is not biased to any particular text because of having more words from it, thus having same number of words from each text will ensure an unbiased model at the end. This function is applied to the processed texts from the previous step and stored in variables.

**DEFINING ANOTHER FUNTION (random_sample) TO ASSIGN 100 WORDS TO A DOCUMENT AND PICK 150 RANDOM DOCUMENTS FROM EACH TEXT**


```python
def random_sample(list_words):
    count = 0
    str = ''
    for word in list_words:
        if count<100:
            str = str + word
            str = str+' '
            count = count + 1
        else:
            str = str+'###'
            count = 0
    strr = str.split('###')
    return random.sample(strr,150)
  
```

**APPLYING THE SECOND ABOVE FUNCTION (random_sample) TO THE PROCESSED TEXTS**


```python
r_str1 = random_sample(words1)
r_str2 = random_sample(words2)
r_str3 = random_sample(words3)
r_str4 = random_sample(words4)
r_str5 = random_sample(words5)
r_str6 = random_sample(words6)
r_str7 = random_sample(words7)
r_str8 = random_sample(words8)
r_str9 = random_sample(words9)
```


```python
print(r_str1[4])
```

    lady must pay subsequent object lament heat suffer walk nothing when i get donwell say knightley could find very odd unaccountable note i send morning message return certainly home till one donwell cry wife my dear mr e donwell you mean crown come meet crown no morrow i particularly want see knightley day account such dreadful broil morning i go field speak tone great ill usage make much bad and find home i assure i please and apology leave message the housekeeper declare know nothing expect very extraordinary and nobody know way go perhaps hartfield perhaps abbey mill perhaps wood 
    
Then we label each of this document obtained from the previous with the corresponding author's name and then we shuffle them, followed by storing them in a single dataframe.

**ASSIGNING THE NAME OF THE AUTHOR OF EACH DOCUMENT TO THE RESPECTIVE DOCUMENT FOR LABELLING THEM**


```python
labeled_names = ([(sent, 'austen') for sent in r_str1] + [(sent, 'bible') for sent in r_str2]+[(sent, 'whitman') for sent in r_str3]+
                 [(sent, 'milton') for sent in r_str4]+[(sent, 'melville') for sent in r_str5]+[(sent, 'edgeworth') for sent in r_str6]+
								         [(sent, 'chesterton') for sent in r_str7]+[(sent, 'shakespeare') for sent in r_str8]+[(sent, 'bryant') for sent in r_str9])
pd.DataFrame(labeled_names)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>cautious think emma advance inch inch hazard n...</td>
      <td>austen</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mr dixon true indeed i think mr dixon she must...</td>
      <td>austen</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mr weston direct whole officiate safely hartfi...</td>
      <td>austen</td>
    </tr>
    <tr>
      <th>3</th>
      <td>spot engage much benefit expect change emma he...</td>
      <td>austen</td>
    </tr>
    <tr>
      <th>4</th>
      <td>lady must pay subsequent object lament heat su...</td>
      <td>austen</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1345</th>
      <td>and i rather think reputation pretty well hamm...</td>
      <td>bryant</td>
    </tr>
    <tr>
      <th>1346</th>
      <td>fell fairy foot but beautiful smile bid come k...</td>
      <td>bryant</td>
    </tr>
    <tr>
      <th>1347</th>
      <td>giant kill after one giant leave never come co...</td>
      <td>bryant</td>
    </tr>
    <tr>
      <th>1348</th>
      <td>she go straight kind sister asylum tell go giv...</td>
      <td>bryant</td>
    </tr>
    <tr>
      <th>1349</th>
      <td>rid hin she get dizzy shure wid lookin fox tai...</td>
      <td>bryant</td>
    </tr>
  </tbody>
</table>
<p>1350 rows × 2 columns</p>
</div>

The Dataframe is then shuffled to avoid any bias and then we perform the word to vector transformation.

**SHUFFLING THE LABELED NAMES AND VIEWING THEM IN A DATAFRAME USING PANDAS**


```python
random.shuffle(labeled_names)
labeled = pd.DataFrame(labeled_names)
```

**NAMING THE COLUMNS OF THE DATAFRAME**


```python
labeled = labeled.rename(columns={0: "Text", 1: "Author"})
labeled.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Text</th>
      <th>Author</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>front behind black group opponent could see li...</td>
      <td>chesterton</td>
    </tr>
    <tr>
      <th>1</th>
      <td>innocent courtesy characteristic insist go las...</td>
      <td>chesterton</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nothing like fix vivid conception peril freque...</td>
      <td>melville</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sword also still sombre dark frock coat hat th...</td>
      <td>chesterton</td>
    </tr>
    <tr>
      <th>4</th>
      <td>rise play neither let u commit fornication com...</td>
      <td>bible</td>
    </tr>
  </tbody>
</table>
</div>



**STORING THE INPUT AND THEIR CORRESPONDING LABELS IN TWO SEPERATE VARIABLES**


```python
X = labeled['Text'].values
y = labeled['Author'].values
```


```python
pd.DataFrame(X).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>front behind black group opponent could see li...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>innocent courtesy characteristic insist go las...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nothing like fix vivid conception peril freque...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sword also still sombre dark frock coat hat th...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>rise play neither let u commit fornication com...</td>
    </tr>
  </tbody>
</table>
</div>



**LABEL ENCODING THE AUTHOR COLUMN** 


```python
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)
```


```python
pd.DataFrame(y).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



**FEATURE ENGINEERING**
Feature transformation is an approach for converting all the textual data into numeric form as the Machine Learning Algorithms work only with numeric data. Since we only have textual data available, the numeric features are extracted by using two different techniques which are Bag-of-Words (BOW) and Term Frequency-Inverse Document Frequency (TF-IDF). These are discussed below.

**TRANSFORMATION OF THE TEXT USING BAG OF WORDS**

```python
count = CountVectorizer(min_df=3, analyzer='word', ngram_range=(1,2), max_features=5000) #CONSIDERING BOTH BIGRAMS AND UNIGRAMS, IGNORING WORDS THAT HAVE A DOCUMENT FREQUENCY OF LESS THAN 3 AND CONSIDERING THE TOP 5000 FEATURES BASED ON FREQUENCY ACROSS THE CORPUS.
bow = count.fit_transform(X)
bow.toarray()
```




    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]])




```python
Bow_feature_names = count.get_feature_names()
```


```python
X_Bow = pd.DataFrame(bow.toarray(), columns=Bow_feature_names) #VIEWING IN FORM OF A DATAFRAME
```

**TRANSFORMATION OF THE TEXT USING TF-IDF**


```python
tf = TfidfVectorizer(analyzer='word',min_df= 3, ngram_range=(1, 2), max_features=5000)
Tfid = tf.fit_transform(X)
Tfid.toarray()
```




    array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]])




```python
tfid_feature_names = tf.get_feature_names()
X_Tfid = pd.DataFrame(Tfid.toarray(), columns=tfid_feature_names)
X_Tfid
```

**USING VARIOUS ML ALGORITHMS FOR MODELING THE PREDICTOR :**

The Machine Learning algorithms we used and compared for this classification problem are:

1. Random Forest
2. Support Vector Machine
3. K-Nearest Neighbor
4. Decision Tree

**Random Forest Classifier with Bag Of Words:**


```python
kf = KFold(n_splits = 10, shuffle = False, random_state=None)
rf_reg = RandomForestClassifier()

scores = []
for i in range(10):
    result = next(kf.split(X_Bow), None)
    X_train = X_Bow.iloc[result[0]]
    X_test = X_Bow.iloc[result[1]]
    y_train = y[result[0]]
    y_test = y[result[1]]
    model = rf_reg.fit(X_train,y_train)
    predictions = rf_reg.predict(X_test)
    scores.append(model.score(X_test,y_test))
print('Scores from each Iteration: ', scores)
RandomForest_Bow = np.mean(scores)
print('Average K-Fold(Random Forest- BOW) Score :' , RandomForest_Bow)
```

    Scores from each Iteration:  [0.8814814814814815, 0.8888888888888888, 0.8814814814814815, 0.8814814814814815, 0.8888888888888888, 0.8814814814814815, 0.8814814814814815, 0.8592592592592593, 0.8814814814814815, 0.8888888888888888]
    Average K-Fold(Random Forest- BOW) Score : 0.8814814814814815
    

**Random Forest with TF-IDF:**


```python
kf = KFold(n_splits = 10, shuffle = False, random_state=None)
rf_reg = RandomForestClassifier()

scores = []
for i in range(10):
    result = next(kf.split(X_Tfid), None)
    X_train = X_Tfid.iloc[result[0]]
    X_test = X_Tfid.iloc[result[1]]
    y_train = y[result[0]]
    y_test = y[result[1]]
    model = rf_reg.fit(X_train,y_train)
    predictions = rf_reg.predict(X_test)
    scores.append(model.score(X_test,y_test))
print('Scores from each Iteration: ', scores)
RandomForest_Tfid = np.mean(scores)
print('Average K-Fold Score(Random Forest- TFid) :' , RandomForest_Tfid)
```

    Scores from each Iteration:  [0.9037037037037037, 0.8814814814814815, 0.8962962962962963, 0.9333333333333333, 0.9111111111111111, 0.8962962962962963, 0.9185185185185185, 0.9037037037037037, 0.9037037037037037, 0.9333333333333333]
    Average K-Fold Score(Random Forest- TFid) : 0.9081481481481483
    

**Support Vector Classifier with Bag Of Words:**


```python
f = KFold(n_splits = 10, shuffle = False, random_state=None)
clf = SVC()

scores = []
for i in range(10):
    result = next(kf.split(X_Bow), None)
    X_train = X_Bow.iloc[result[0]]
    X_test = X_Bow.iloc[result[1]]
    y_train = y[result[0]]
    y_test = y[result[1]]
    model = clf.fit(X_train,y_train)
    predictions = clf.predict(X_test)
    scores.append(model.score(X_test,y_test))
print('Scores from each Iteration: ', scores)
SVM_Bow = np.mean(scores)
print('Average K-Fold Score(SVM- BOW) :' , SVM_Bow)

```

    Scores from each Iteration:  [0.9185185185185185, 0.9185185185185185, 0.9185185185185185, 0.9185185185185185, 0.9185185185185185, 0.9185185185185185, 0.9185185185185185, 0.9185185185185185, 0.9185185185185185, 0.9185185185185185]
    Average K-Fold Score(SVM- BOW) : 0.9185185185185183
    

**Support Vector Classifier with TF-IDF:**


```python
kf = KFold(n_splits = 10, shuffle = False, random_state=None)
clf = SVC()

scores = []
for i in range(10):
    result = next(kf.split(X_Tfid), None)
    X_train = X_Tfid.iloc[result[0]]
    X_test = X_Tfid.iloc[result[1]]
    y_train = y[result[0]]
    y_test = y[result[1]]
    model = clf.fit(X_train,y_train)
    predictions = clf.predict(X_test)
    scores.append(model.score(X_test,y_test))
print('Scores from each Iteration: ', scores)
SVM_Tfid = np.mean(scores)
print('Average K-Fold Score(SVM- TFid) :' , SVM_Tfid)

```

    Scores from each Iteration:  [0.9481481481481482, 0.9481481481481482, 0.9481481481481482, 0.9481481481481482, 0.9481481481481482, 0.9481481481481482, 0.9481481481481482, 0.9481481481481482, 0.9481481481481482, 0.9481481481481482]
    Average K-Fold Score(SVM- TFid) : 0.9481481481481481
    

**Decision Tree Classifier using Bag Of Words:**


```python
kf = KFold(n_splits = 10, shuffle = False, random_state=None)
dtc = DecisionTreeClassifier()

scores = []
for i in range(10):
    result = next(kf.split(X_Bow), None)
    X_train = X_Bow.iloc[result[0]]
    X_test = X_Bow.iloc[result[1]]
    y_train = y[result[0]]
    y_test = y[result[1]]
    model = dtc.fit(X_train,y_train)
    predictions = dtc.predict(X_test)
    scores.append(model.score(X_test,y_test))
print('Scores from each Iteration: ', scores)
DecisionTree_Bow = np.mean(scores)
print('Average K-Fold Score(Decision Tree- BOW) :' , DecisionTree_Bow)

```

    Scores from each Iteration:  [0.6888888888888889, 0.6888888888888889, 0.7185185185185186, 0.6962962962962963, 0.6518518518518519, 0.6888888888888889, 0.674074074074074, 0.7111111111111111, 0.6888888888888889, 0.6592592592592592]
    Average K-Fold Score(Decision Tree- BOW) : 0.6866666666666668
    

**Decision Tree Classifier using TF-IDF:**


```python
kf = KFold(n_splits = 10, shuffle = False, random_state=None)
dtc = DecisionTreeClassifier()

scores = []
for i in range(10):
    result = next(kf.split(X_Tfid), None)
    X_train = X_Tfid.iloc[result[0]]
    X_test = X_Tfid.iloc[result[1]]
    y_train = y[result[0]]
    y_test = y[result[1]]
    model = dtc.fit(X_train,y_train)
    predictions = dtc.predict(X_test)
    scores.append(model.score(X_test,y_test))
print('Scores from each Iteration: ', scores)
DecisionTree_Tfid = np.mean(scores)
print('Average K-Fold Score(Decision Tree- TFid) :' , DecisionTree_Tfid)
```

    Scores from each Iteration:  [0.6962962962962963, 0.725925925925926, 0.7333333333333333, 0.725925925925926, 0.725925925925926, 0.7111111111111111, 0.6962962962962963, 0.7037037037037037, 0.725925925925926, 0.7037037037037037]
    Average K-Fold Score(Decision Tree- TFid) : 0.7148148148148149
    

**K-NN Classifier using Bag Of Words**


```python
kf = KFold(n_splits = 10, shuffle = False, random_state=None)
neigh = KNeighborsClassifier(n_neighbors=3)

scores = []
for i in range(10):
    result = next(kf.split(X_Bow), None)
    X_train = X_Bow.iloc[result[0]]
    X_test = X_Bow.iloc[result[1]]
    y_train = y[result[0]]
    y_test = y[result[1]]
    model = neigh.fit(X_train,y_train)
    predictions = neigh.predict(X_test)
    scores.append(model.score(X_test,y_test))
print('Scores from each Iteration: ', scores)
KNN_Bow = np.mean(scores)
print('Average K-Fold Score(KNN- BOW) :' , KNN_Bow)
```

    Scores from each Iteration:  [0.43703703703703706, 0.43703703703703706, 0.43703703703703706, 0.43703703703703706, 0.43703703703703706, 0.43703703703703706, 0.43703703703703706, 0.43703703703703706, 0.43703703703703706, 0.43703703703703706]
    Average K-Fold Score(KNN- BOW) : 0.437037037037037
    

**K-NN Classifier using TF-IDF**


```python
kf = KFold(n_splits = 10, shuffle = False, random_state=None)
neigh = KNeighborsClassifier(n_neighbors=3)

scores = []
for i in range(10):
    result = next(kf.split(X_Tfid), None)
    X_train = X_Tfid.iloc[result[0]]
    X_test = X_Tfid.iloc[result[1]]
    y_train = y[result[0]]
    y_test = y[result[1]]
    model = neigh.fit(X_train,y_train)
    predictions = neigh.predict(X_test)
    scores.append(model.score(X_test,y_test))
print('Scores from each Iteration: ', scores)
KNN_Tfid = np.mean(scores)
print('Average K-Fold Score(KNN- TFid) :' , KNN_Tfid)
```

    Scores from each Iteration:  [0.8518518518518519, 0.8518518518518519, 0.8518518518518519, 0.8518518518518519, 0.8518518518518519, 0.8518518518518519, 0.8518518518518519, 0.8518518518518519, 0.8518518518518519, 0.8518518518518519]
    Average K-Fold Score(KNN- TFid) : 0.8518518518518519
    

**Visualizing the accuracies of the model (TF-IDF)**


```python
entries = [('RandomForest', RandomForest_Tfid),('SVM', SVM_Tfid), ('Decision Tree', DecisionTree_Tfid), ('KNN', KNN_Tfid)]
cv_df = pd.DataFrame(entries, columns=['model_name', 'accuracy'])
import seaborn as sns
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()
```


![png](/images/Text_Classification_NLTK_files/Text_Classification_NLTK_67_0.png)


**Visualizing the accuracies of the model (Bag Of Words)**


```python
entries = [('RandomForest', RandomForest_Bow),('SVM', SVM_Bow), ('Decision Tree', DecisionTree_Bow), ('KNN', KNN_Bow)]
cv_df = pd.DataFrame(entries, columns=['model_name', 'accuracy'])
import seaborn as sns
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()
```

**MODEL EVALUATION**
**SVC  (TF-IDF)**


```python
model = SVC()

X_train,X_test,y_train,y_test = train_test_split(X_Tfid,y, test_size=0.30, random_state=5)
```


```python
model.fit(X_train,y_train)
```




    SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)




```python
y_pred = model.predict(X_test)
```


```python
print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       1.00      0.97      0.99        39
               1       0.94      1.00      0.97        34
               2       0.96      0.96      0.96        45
               3       1.00      0.94      0.97        47
               4       0.98      0.98      0.98        51
               5       1.00      0.94      0.97        49
               6       0.96      0.96      0.96        53
               7       1.00      0.98      0.99        44
               8       0.88      1.00      0.93        43
    
        accuracy                           0.97       405
       macro avg       0.97      0.97      0.97       405
    weighted avg       0.97      0.97      0.97       405
    
    


```python
print(confusion_matrix(y_test, y_pred))
```

    [[38  0  0  0  0  0  0  0  1]
     [ 0 34  0  0  0  0  0  0  0]
     [ 0  1 43  0  1  0  0  0  0]
     [ 0  0  0 44  0  0  1  0  2]
     [ 0  0  1  0 50  0  0  0  0]
     [ 0  0  1  0  0 46  0  0  2]
     [ 0  1  0  0  0  0 51  0  1]
     [ 0  0  0  0  0  0  1 43  0]
     [ 0  0  0  0  0  0  0  0 43]]
    

**Visualizing the Confusion Matrix:**


```python
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```


![png](/images/Text_Classification_NLTK_files/Text_Classification_NLTK_77_0.png)

The accuracy for SVM algorithm (for both BOW and TF-IDF transformations) was the highest compared to all the other models, we have selected it as our champion model.

**ERROR ANALYSIS:**

Continuing with our best model (i.e. the Linear SVM), we looked at the confusion matrix in order to see the discrepancies between predicted and actual labels if any. And from the confusion matrix, we observed that most of the predictions made by the model were correct (as vast majority of the predictions were present at the diagonal). But there were a few misclassification and thus we tried to find the possible reason behind these misclassifications.

```python
encoding = [(0,'austen'),(1,'bible'),(2,'whitman'),(3,'milton'),(4,'melville'),(5, 'edgeworth'),(6,'chesterton'),(7,'shakespear'),(8,'bryant')]
auth = pd.DataFrame(encoding)
auth = auth.rename(columns={0: "Label", 1: "Author"})
print(auth)
```


```python
for predicted in auth.Label:
    for actual in auth.Label:
        if predicted != actual and conf_mat[actual, predicted] >= 1:
            print("'{}' predicted as '{}' : {} examples.".format(y_test[actual],y_pred[predicted], conf_mat[actual, predicted]))
        display(labeled.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['Author', 'Text']])
      print('')
```
