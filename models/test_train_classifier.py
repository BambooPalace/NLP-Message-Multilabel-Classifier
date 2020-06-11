import warnings
warnings.filterwarnings("ignore")# for warning when model handle sparse matrix
                        
import sys
# import libraries
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score
from joblib import dump,load

nltk.download('punkt') #tokenizer
nltk.download('stopwords') 
nltk.download('wordnet')# lemmatizer

def load_data(database_filepath):
    #load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('SELECT * FROM messages', engine)
    X = df.message.values
    Y = df.iloc[:,2:].values
    labels=df.columns[2:].tolist()
    return X,Y,labels


def tokenize(text):
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize text
    tokens = word_tokenize(text)
    # lemmatize and remove stop words
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens


def build_model():
    #build a pipeline including feature extraction and modelling (Naive Baiyes)
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(BernoulliNB()))
     ])
    #set gridsearch parameters 
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.75, 1.0),
        'tfidf__use_idf': (True, False)
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3)
    return cv


def evaluate_model(model, X_test, Y_test,category_names):
    Y_preds=model.predict(X_test)
    n=len(category_names)
    f1_list=[f1_score(Y_test[:,i], Y_preds[:,i],average='weighted')for i in range(n)]
    print('model mean f1_score on test set is: ', sum(f1_list)/n)


def save_model(model, model_filepath):
    dump(model, model_filepath)

# below are my functions for testing
def load_model():
    # test show error if use sklearn.externals.joblib to unload
    model = load('models/classifier.pkl') 
    return model

def display_category(text,model):
    # display prediction of a message in dataframe format
    cat=model.predict(text)[0]
    df=pd.DataFrame({'prediction': cat}, index=labels)
    df= df[df.prediction!=0]
    print(df)

def test_classifier(text):
    #load model and run on a test text sample

    model=load_model()
    df=display_category(text,model)
    print(df)


# python models/test_train_classifier.py data/DisasterResponse.db models/classifier.pkl
if __name__ == '__main__':
    _,_,labels=load_data('data/DisasterResponse.db')
    text=["I hear thunder"]        
    print('Classifier for below message: \n\t'+ text[0])
    test_classifier(text)

  
    