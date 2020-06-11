# import warnings
# warnings.filterwarnings("ignore")# for warning when model handle sparse matrix
                        
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
from sklearn.metrics import classification_report,f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
#The classifier- ComplementNB need scikit-learn 0.20+ to run, need to update workspace 
from sklearn.naive_bayes import ComplementNB
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
        ('clf', MultiOutputClassifier(ComplementNB()))
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
    f1_list_weighted=[]
    for i in range(n):
        print(classification_report(Y_test[:,i], Y_preds[:,i]),('for label:'+category_names[i].upper()))    

    
def save_model(model, model_filepath):
    dump(model, model_filepath)

    
def test_model(text,model, labels):
    # display prediction of a message in dataframe format
    cat=model.predict(text)[0]
    df=pd.DataFrame({'classification': cat}, index=labels)
    col= df[df.classification!=0].index.values
    print(col)    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
        

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test,category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
        
        print('Testing model')
        text=["I hear thunder"]        
        print('Classifier for below message: \n\t'+ text[0])
        test_model(text, model,category_names)       

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()