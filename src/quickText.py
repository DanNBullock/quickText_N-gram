"""
The following is a set of functions designed to perform some quick and dirty text analysis.
Broadly speaking, the goal is to ingest a document (e.g. docx, pdf, txt), extract the text,
perform lemmaization, apply a stopword list (from a csv, if provided), and then return a count
of the most common n-grams (e.g. 1-grams, 2-grams, 3-grams, etc.) in the document.


"""

def ingestDocToText(docPath):
    """
    This function takes a path to a document (docx, pdf, txt) and returns the text as a string.
    """
    import docx2txt
    import PyPDF2
    import textract
    import os

    # Get the file extension
    _, file_extension = os.path.splitext(docPath)

    # If the file is a docx, use docx2txt
    if file_extension == '.docx':
        text = docx2txt.process(docPath)

    # If the file is a pdf, use PyPDF2
    elif file_extension == '.pdf':
        pdfFileObj = open(docPath, 'rb')
        pdfReader = PyPDF2.PdfReader(pdfFileObj)
        numPages = len(pdfReader.pages)
        text = ''
        for i in range(numPages):
            pageObj = pdfReader.pages[i]
            text += pageObj.extract_text()

    # If the file is a txt just read it in
    elif file_extension == '.txt':
        text = textract.process(docPath, method='tesseract', language='eng')


    # If the file is not a docx, pdf, or txt, return an error
    else:
        print('Error: File type not supported.')
        return None

    return text

def lemmatizeText(inputText):
    """
    This function takes in the text from an entire document, and lemmatizes the text, converting
    varied forms of the same word to the same lemma, lowercasing all words, and removing punctuation.
   
    
    """

    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    import string

    # Tokenize the text
    tokenizedText = word_tokenize(inputText)

    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatize the text
    lemmatizedText = [lemmatizer.lemmatize(word.lower()) for word in tokenizedText if word not in string.punctuation]

    return lemmatizedText

def applyStopwords(inputText, nltkStopwords=True, customStopwords=None):
    """
    This function takes in the text from an entire document, and applies a list of stopwords to it.
    If nltkStopwords is set to true, the function will use the nltk stopwords list.  Additionally,
    if customStopwords is not none (e.g. a path to a csv file), the function will also apply the
    custom stopwords list.
    """
    
    import nltk
    from nltk.corpus import stopwords
    import pandas as pd

    # If nltkStopwords is true, use the nltk stopwords list
    if nltkStopwords:
        nltkStopwords = stopwords.words('english')
    else:
        nltkStopwords = []

    # If customStopwords is not none, use the custom stopwords list
    if customStopwords is not None:
        customStopwords = pd.read_csv(customStopwords)
        customStopwords = customStopwords['stopwords'].tolist()
    else:
        customStopwords = []

    # Combine the nltk and custom stopwords lists
    stopwordsList = nltkStopwords + customStopwords

    # Apply the stopwords list to the text.  Assume the input inputText is a single string
    # and not a list of strings
    textWithoutStopwords = [word for word in inputText.split() if word not in stopwordsList]

    # Convert the text back to a string
    textWithoutStopwords = ' '.join(textWithoutStopwords)

    return textWithoutStopwords

def getNGrams(inputText, n):
    """
    This function takes in the text from an entire document, and returns a list of n-grams.
    Ordered by frequency.  The ngrams and their frequencies are returned in a pandas dataframe.
    """
    from nltk import ngrams
    from collections import Counter
    import pandas as pd

    # assume that the input is a single string, separated by spaces
    # split the string into a list of words
    inputText = inputText.split()

    # Get the n-grams
    nGrams = ngrams(inputText, n)

    # convert the n-grams to a list in which the n-gram elements have been joined
    nGrams = [' '.join(gram) for gram in nGrams]
    nGrams = list(nGrams)

    # Count the n-grams
    nGramCounts = Counter(nGrams)

    # Convert the n-grams and their frequencies to a pandas dataframe
    nGramCounts = pd.DataFrame.from_dict(nGramCounts, orient='index').reset_index()

    # rename the columns
    nGramCounts.columns = ['ngram', 'frequency']
    
    return nGramCounts

def aggregateNgramLists(nGramLists):
    """
    This function takes in a list of n-gram lists along with frequency information (provided 
    via a pandas dataframe with columns ['ngram', 'frequency']), and aggregates and de-duplicates
    the n-grams across the lists, for example reducing the count of a 1-gram that appears in a 2-gram
    by the count of the 2-gram (in this way preferencing the longer n-gram).  The function returns
    a pandas dataframe with columns ['ngram', 'frequency'].
    """
    import pandas as pd

    # Concatenate the n-gram lists
    nGramLists = pd.concat(nGramLists)

    # now that the n-grams have been collected into a single dataframe, we need to remove 
    # instances of double counting, preferencing the longer n-gram
    # we'll do this by iterating through the n-grams, starting with the shortest n-grams
    # and reducing the count that n-gram by the count of matching longer n-grams
    
    # first we'll sort the n-grams by length, using split to count the number of words in the n-gram
    nGramLists['length'] = nGramLists['ngram'].apply(lambda x: len(x.split()))
    nGramLists = nGramLists.sort_values(by='length', ascending=True)
    # now we'll begin iterating through the n-grams
    for index, row in nGramLists.iterrows():
        # the nGramLists dataframe has been sorted such that the shortest n-grams are first
        # so we'll start by looking for n-grams that are longer than the current n-gram
        # we won't need to look at n-grams that are shorter than the current n-gram
        # because we are processing the n-grams in order of length
        # we'll use the length column to filter the dataframe
        longerNGrams = nGramLists[nGramLists['length'] > row['length']]
        # now we can see if the current n-gram is a substring of any of the longer n-grams
        # we'll use the contains method to check for this
        longerNgramsContainingCurrentNgram = longerNGrams[longerNGrams['ngram'].str.contains(row['ngram'])]
        # if there are any longer n-grams that contain the current n-gram, we'll reduce the count of the current n-gram
        # by the count of the longer n-grams
        if len(longerNgramsContainingCurrentNgram) > 0:
            nGramLists.loc[index, 'frequency'] = row['frequency'] - longerNgramsContainingCurrentNgram['frequency'].sum()
    # now that we've reduced the counts of the shorter n-grams, we can remove the length column
    nGramLists = nGramLists.drop(columns='length')
    # finally, we'll remove any n-grams that have a frequency of 0
    nGramLists = nGramLists[nGramLists['frequency'] > 0]

    return nGramLists

def getNgramFrequencies(docPath, nltkStopwords=True, customStopwords=None):
    """
    This function takes in the path to a document, and returns a list of n-grams (from 1 to 3) and their frequencies.
    The n-grams and their frequencies are returned in a pandas dataframe.
    """
    # Get the text from the document
    import pandas as pd

    inputText = ingestDocToText(docPath)

    import nltk
    # Lemmatize the text
    lemmatizedText = lemmatizeText(inputText)

    # convert the lemmatizedText, which is a list, back to a single string
    lemmatizedText = ' '.join(lemmatizedText)

    # Apply the stopwords list to the text
    textWithoutStopwords = applyStopwords(lemmatizedText, nltkStopwords, customStopwords)


    # get n-grams for n = 1, 2, and 3
    unigrams = getNGrams(textWithoutStopwords, 1)
    bigrams = getNGrams(textWithoutStopwords, 2)
    trigrams = getNGrams(textWithoutStopwords, 3)
    # combine the n-grams into a single list with frequency information
    nGrams = [unigrams, bigrams, trigrams]
    # de-duplicate the n-grams
    # nGrams = aggregateNgramLists(nGrams)
    # stack the dataframes into a single dataframe
    nGrams = pd.concat(nGrams)
    # sort the n-grams by frequency
    nGrams = nGrams.sort_values(by='frequency', ascending=False)
    # reset the index
    nGrams = nGrams.reset_index(drop=True)    
    return nGrams

"""
Here we have the boilerplate to set paths and run the code above
"""
pathToDoc='C:\\Users\\iisda\\Documents\\work\\08-2022-OSTP-Public-Access-Memo.pdf'
pathToStopWords=None
nltkStopwords=True
savePath='C:\\Users\iisda\\Documents\\work\\OSTP_Memo_nGrams.csv'
# now run it
nGrams = getNgramFrequencies(pathToDoc, nltkStopwords, pathToStopWords)


nGrams.to_csv(savePath, index=False)