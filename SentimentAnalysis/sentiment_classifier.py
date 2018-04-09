'''
This script performs the basic process for applying a machine learning
algorithm to a dataset using Python libraries.

The four steps are:
   1. Download a dataset (using pandas)
   2. Process the numeric data (using numpy)
   3. Train and evaluate learners (using scikit-learn)
   4. Plot and compare results (using matplotlib)


The data is downloaded from URL, which is defined below. As is normal
for machine learning problems, the nature of the source data affects
the entire solution. When you change URL to refer to your own data, you
will need to review the data processing steps to ensure they remain
correct.
'''

URL = r'C:\Users\vaibh\Source\Repos\DMTM_UIC\project_2_train\data 1_train.csv'

# Uncomment this call when using matplotlib to generate images
# rather than displaying interactive UI.
#import matplotlib
#matplotlib.use('Agg')

from pandas import read_table
import numpy as np
import matplotlib.pyplot as plt

try:
    # [OPTIONAL] Seaborn makes plots nicer
    import seaborn
except ImportError:
    pass

# =====================================================================

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
   
# =====================================================================
# We will calculate the P-R curve for each classifier
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score
def execute(clf, clf_name, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    score = f1_score(y_test, pred, average='weighted')
    acc = accuracy_score(y_test, pred)

    # Generate the P-R curve
    y_prob = clf.decision_function(X_test)
    precision, recall, avg = get_per_class_pr_re_and_avg(y_test, y_prob)

    # Include the score in the title
    print('{} (F1 score={:.3f}, Accuracy={:.4f})'.format(clf_name, score, acc))
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(avg["micro"]))
    return precision, recall, avg

def get_per_class_pr_re_and_avg(Y_test, y_score):
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(Y_test.shape[1]):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
        y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                         average="micro")
    return precision, recall, average_precision

def plot_avg_p_r_curves(precision, recall, average_precision):
    plt.figure()
    plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'
                .format(average_precision["micro"]))

def plot_per_class_p_r_curves(precision, recall, average_precision, classes):
    from itertools import cycle
    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    plt.figure(figsize=(7, 8))
    lines = []
    labels = []
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))

    for i, color in zip(range(len(classes)), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(classes[i], average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    plt.show()

def plot(results):
    '''
    Create a plot comparing multiple learners.

    `results` is a list of tuples containing:
        (title, precision, recall)
    
    All the elements in results will be plotted.
    '''

    # Plot the precision-recall curves

    fig = plt.figure(figsize=(6, 6))
    fig.canvas.set_window_title('Classifying data from ' + URL)

    for label, precision, recall in results:
        plt.plot(recall, precision, label=label)

    plt.title('Precision-Recall Curves')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.legend(loc='lower left')

    # Let matplotlib improve the layout
    plt.tight_layout()

    # ==================================
    # Display the plot in interactive UI
    plt.show()

    # To save the plot to an image file, use savefig()
    #plt.savefig('plot.png')

    # Open the image file with the default image viewer
    #import subprocess
    #subprocess.Popen('plot.png', shell=True)

    # To save the plot to an image in memory, use BytesIO and savefig()
    # This can then be written to any stream-like object, such as a
    # file or HTTP response.
    #from io import BytesIO
    #img_stream = BytesIO()
    #plt.savefig(img_stream, fmt='png')
    #img_bytes = img_stream.getvalue()
    #print('Image is {} bytes - {!r}'.format(len(img_bytes), img_bytes[:8] + b'...'))

    # Closing the figure allows matplotlib to release the memory used.
    plt.close()


# =====================================================================


if __name__ == '__main__':
    # Import some classifiers to test
    from sklearn.svm import LinearSVC, NuSVC
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.multiclass import OneVsRestClassifier

    # Download the data set from URL
    print("Downloading data from {}".format(URL))
    frame = download_data()

    # Process data into feature and label arrays
    print("Processing {} samples with {} attributes".format(len(frame.index), len(frame.columns)))
    X_train, X_test, y_train, y_test = get_features_and_labels(frame, [-1,0,1], True)

    # Evaluate multiple classifiers on the data
    print("Evaluating classifiers")
    classifiers = []
    classifiers.append([OneVsRestClassifier(LinearSVC()), "OVR LinearSVC"])
    #classifiers.append([LinearSVC(C=1), "LinearSVC"])
    #classifiers.append([NuSVC(kernel='rbf', nu=0.5, gamma=1e-3), "NuSVC"])
    #classifiers.append([AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm='SAMME.R'), "AdaBoost"])

    results = []

    for clf in classifiers:
        results.append(execute(clf[0], clf[1], X_train, X_test, y_train, y_test))

    # Display the results
    print("Plotting the results")
    #plot(results)
    for r in results:
        plot_avg_p_r_curves(r[0], r[1], r[2])
        plot_per_class_p_r_curves(r[0], r[1], r[2], [-1,0,1])   # Update classes
