import numpy as np
import string
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from pandas import read_table

def download_data(url):
    '''
    Downloads the data for this script into a pandas DataFrame.
    '''
    frame = read_table(url,
        
    # Uncomment if the file needs to be decompressed
    #compression='gzip',
    #compression='bz2',

    # Specify the file encoding
    # Latin-1 is common for data from US sources
    #encoding='latin-1',
    encoding='utf-8',  # UTF-8 is also common

    # Specify the separator in the data
    sep=',',            # comma separated values
    #sep='\t', # tab separated values
    #sep=' ', # space separated values

    # Ignore spaces after the separator
    skipinitialspace=True,

    # Generate row labels from each row number
    index_col=None,
    #index_col=0, # use the first column as row labels
    #index_col=-1, # use the last column as row labels

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

from nltk.parse.corenlp import CoreNLPDependencyParser
dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')
def parse_dependencies(text, aspect):
    '''
    Uses Stanford CoreNLP (https://stanfordnlp.github.io/CoreNLP/) to parse
    and extract sentiment words for aspect using dependency graph.
    Stanford NLP Parser demo page: http://nlp.stanford.edu:8080/corenlp/process
    See also: http://nlp.stanford.edu:8080/sentiment/rntnDemo.html
    and: http://www.nltk.org/api/nltk.parse.html#nltk.parse.corenlp.CoreNLPDependencyParser
    '''
    parse, = dep_parser.raw_parse(text)
    #print(text)
    extracted_deps = []
    extract_dependencies(list(parse.triples()), aspect, extracted_deps)
    return extracted_deps

def extract_dependencies(parsed_triples, aspect, extracted_deps, depth=0, max_depth=-1):
    '''
    Performs IDDFS to extract dependencies of aspect.
    Use max_depth = 1, for just direct dependencies, -1 for all. 'depth' = current depth.
    '''
    if (max_depth != -1 and depth == max_depth) or aspect == "" or aspect == None:
        return
    depth += 1
    dependencies_to_extract = ["nsubj", "advmod", "amod", "nmod", "advcl", "acl:relcl", "acl:relcl", "aux", "neg", "conj"]
    pos = ["NN", "NNS", "JJ"]   # POS to consider

    # Handle multi-word aspect
    aspect = aspect.split(" ")

    # Used extra filter in comprehension for increased efficiency.
    for governor, dep, dependent in [(g,d,dp) for (g,d,dp) in parsed_triples if g[0] in aspect or dp[0] in aspect]:
        #[(g,d,dp) for (g,d,dp) in parse.triples() if aspect in g or aspect in dp]:
        if dep in dependencies_to_extract:
            aspect_d = None
            if governor[0] in aspect and dependent[1] in pos:   # Eg. ('horrible', 'JJ') nsubj ('staff', 'NN')
                # Extract dependent and recurse
                aspect_d = dependent[0]
                if not aspect_d in extracted_deps:
                    extracted_deps.append(aspect_d)
            elif dependent[0] in aspect and governor[1] in pos:
                # Extract governor and recurse
                aspect_d = governor[0]
                if not aspect_d in extracted_deps:
                    extracted_deps.append(aspect_d)
            if aspect_d != None:
                if (governor, dep, dependent) in parsed_triples:
                    parsed_triples.remove((governor, dep, dependent))
                extract_dependencies(parsed_triples, aspect_d, extracted_deps, depth, max_depth)
    return


def transform_features_and_labels(frame, classes=None, binarize=False):
    '''
    Extracts aspect dependencies, vectorizes and Transforms the input data 
    and returns numpy arrays for training and testing inputs and targets.
    If 'binarize=True', classes must be supplied.
    '''
    
    arr = np.array(frame)
    corpus = []

    #print("Extracting aspect dependencies..")
    print("POS tagging and lemmatizing..")
    for d in np.take(arr, [1,2], axis=1):
        #parsed_text = parse_out_text(d)
        #aspect_dependencies = parse_dependencies(d[0].replace("[comma]",","), d[1])

        # Add aspect as well
        #aspect_dependencies.append(d[1])

        # Lemmatize (Document->Sentences->Tokens->POS->Lemmas)
        corpus.append(lemmatize(d[0]))  #" ".join(aspect_dependencies)

    # Vectorize (TODO: Try WordToVec)
    print("Vectorizing..")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    y = np.array(arr[:, -1], dtype=np.float)  # Use the last column as the target value

    if binarize:
        from sklearn.preprocessing import label_binarize
        # Update class labels here
        y = label_binarize(y, classes)

    return X, y

import nltk
#nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
def lemmatize(text):
    #POS tag and lemmatize text
    # See: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    tokens = word_tokenize(text)
    pos_tagged = nltk.pos_tag(tokens)

    return " ".join([lmtzr.lemmatize(w[0], pos=get_wordnet_pos(w[1])) for w in pos_tagged])

from nltk.corpus import wordnet
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN     
