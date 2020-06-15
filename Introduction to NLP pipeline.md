##Natural Language Processing Pipelines
**In this lesson, you'll be introduced to some of the steps involved in a NLP pipeline:**

###1.Text Processing

- Cleaning: to remove irrelevant items, such as HTML tags

> an example of cleaning text data from a popular source - the web. Helpful tools in working with this data, includes the requests library, regular expressions, and Beautiful Soup.

- Normalizing: by converting to all lowercase and removing punctuation

- Tokenization: **Splitting** text into words or tokens

- Removing stop words: words that are too common, and no meaning

> i.e. pronouns（you, us, him）, prepositions(in, at, on), articles(a,an,the,these), preverbs(too, such, better), and the list goes on.

- POS tagging & Named entities: Identifying different **parts of speech and named entities recognition**

> Note: Part-of-speech tagging using a predefined grammar like this is a simple, but limited, solution. It can be very tedious and error-prone for a large corpus of text, since you have to account for all possible sentence structures and tags!

There are other more advanced forms of POS tagging that can learn sentence structures and tags from given data, including Hidden Markov Models (HMMs) and Recurrent Neural Networks (RNNs).


- Stemming and lemmatization: Converting words into their dictionary forms

> stemming:change/changing/changed/changes/changer => **chang**
> lemmatization:change/changing/changed/changes/changer => **change**

###2.Feature Extraction
In order to feed text data into a statistical or a machine learning model.

> 
- If you want to use a **graph based model** to extract insights,
you may want to represent your words as 
symbolic nodes with relationships between them like WordNet.
For statistical models however,
- If you're trying to perform a **document level** task,
such as **spam detection or sentiment analysis** ,
you may want to use a per document representations such as **bag-of-words or doc2vec**.
- If you want to work with **individual words and phrases**
such as for **text generation or machine translation,**
you'll need a word level representation such as **word2vec or glove.**

- Bag of Words
- TF-IDF (term frequency in document frequency)
- Word Embeddings: 
e.g. DL method: represent word similarity in vector/spatial form

###3.Modeling


