
### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Instructions](#instructions)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The code run with Python3 and some additional packages beside Anaconda pre-installations, i.e. wordcloud and plotly. The machine learning model for NLP is ComplementNB, which is available for scikit-learn 0.20+.

## Project Motivation<a name="motivation"></a>

This project takes in 25000+ disaster messagess to build a classifier for categorizing new messages. 
The project is done in three steps:
1. ETL pipeline: extract raw data, transform/clean data, load wrangled data to database.
2. NLP ML pipeline: read and process text data from database, train and save the machine learning model.
3. Flask Web App: load trained model to classify new messages and display analysis data on webpages, using Flask framework.


## File Descriptions <a name="files"></a>

All app data is in  current *Classifier_App* folder, project code structure is as below: 

\- app<br>
| - template<br>
| |- master.html  # main page of web app<br>
| |- go.html  # classification result page of web app<br>
|- run.py  # Flask file that runs app<br>

\- data<br>
|- disaster_categories.csv  # data to process <br>
|- disaster_messages.csv  # data to process<br>
|- process_data.py<br>
|- InsertDatabaseName.db   # database to save clean data to<br>

\- models<br>
|- train_classifier.py<br>
|- classifier.pkl  # saved model         <br>

Other folders contain preparation data and knowledge files.

## Intructions to run app<a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.
    - update scikit-learn to 0.20+ 

    - To run ETL pipeline that cleans data and stores in database, in command line:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

    - To run ML pipeline that trains classifier and saves, in command line:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`


2. Run the following command in the app's directory to run your web app.
    - install wordcloud and plotly before run
    - in command line:
    `python app/run.py`
 

3. Go to https://SPACEID-3001.SPACEDOMAIN to retrieve app.


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

For the sharing spirit of internet, you are free to use it following License [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/).

