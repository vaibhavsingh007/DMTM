import numpy as np
from pandas import read_table

URL = r'C:\Users\vaibh\Source\Repos\DMTM_UIC\project_2_train\data 1_train.csv'

def download_data():
    '''
    Downloads the data for this script into a pandas DataFrame.
    '''
    frame = read_table(
        URL,
        
        # Uncomment if the file needs to be decompressed
        #compression='gzip',
        #compression='bz2',

        # Specify the file encoding
        # Latin-1 is common for data from US sources
        #encoding='latin-1',
        encoding='utf-8',  # UTF-8 is also common

        # Specify the separator in the data
        sep=',',            # comma separated values
        #sep='\t',          # tab separated values
        #sep=' ',           # space separated values

        # Ignore spaces after the separator
        skipinitialspace=True,

        # Generate row labels from each row number
        index_col=None,
        #index_col=0,       # use the first column as row labels
        #index_col=-1,      # use the last column as row labels

        # Generate column headers row from each column number
        #header=None,
        header=0,          # use the first line as headers

        # Use manual headers and skip the first row in the file
        #header=0,
        #names=['col1', 'col2', ...],
    )

    # Return a subset of the columns
    #return frame[['col1', 'col4', ...]]

    # Return the entire frame
    return frame


# =====================================================================
import string
from nltk.stem.snowball import SnowballStemmer
def parse_out_text(text_string):
    # Throw away punctuations
    # Remove [comma], tabs and double spaces
    text_string = text_string.replace("[comma]"," ")
    text_string = text_string.replace("\t"," ")
    text_string = text_string.replace("  ", " ")
    text_string = text_string.translate(''.maketrans("", "", string.punctuation))
    stemmer = SnowballStemmer("english")
    words = " ".join([stemmer.stem(w.strip()) for w in text_string.split(" ")])
    return words

from sklearn.feature_extraction.text import TfidfVectorizer
def get_features_and_labels(frame, classes=None, binarize=False):
    '''
    Vectorizes and Transforms the input data and returns numpy arrays for
    training and testing inputs and targets.
    If 'binarize=True', classes must be supplied.
    '''

    # Replace missing values with 0.0, or we can use
    # scikit-learn to calculate missing values (below)
    #frame[frame.isnull()] = 0.0

    arr = np.array(frame)

    corpus = []
    for d in arr[:,1]:
        parsed_text = parse_out_text(d)
        corpus.append(parsed_text)

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(corpus)
    y = np.array(arr[:, -1], dtype=np.float)  # Use the last column as the target value

    if binarize:
        from sklearn.preprocessing import label_binarize
        # Update class labels here
        y = label_binarize(y, classes)
    
    # Use 80% of the data for training; test against the rest
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Return the training and test sets
    return X_train, X_test, y_train, y_test
