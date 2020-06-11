import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # load two datasets and merge them into one
    messages = pd.read_csv(messages_filepath, index_col='id')
    categories = pd.read_csv(categories_filepath,index_col='id')
    df = pd.concat([messages,categories], axis=1)
    return df


def clean_data(df):
    ''' clean categories column of data'''
    #from column values, extract column names
    categories = df.categories.str.split(';', expand=True)
    categories.columns=categories.iloc[1].str[:-2].values
    # extract numerical value
    for column in categories:
        categories[column] = pd.to_numeric(categories[column].str[-1])
    # drop the original & old categories column from `df`
    df.drop(['categories','original'], axis=1, inplace=True)
    # merge processed `categories` dataframe to original dataframe 
    df = pd.concat([df,categories], axis=1)
    # given data has no missing values, so handling missing values here
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df
    
def save_data(df, database_filename):
    #save df to sqlite database
    engine=create_engine('sqlite:///'+database_filename)
    df.to_sql('messages', engine, if_exists='replace', index=False)
    pass


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

        
if __name__ == '__main__':
    main()

    