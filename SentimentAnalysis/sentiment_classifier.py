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

# Uncomment this call when using matplotlib to generate images
# rather than displaying interactive UI.
#import matplotlib
#matplotlib.use('Agg')

import numpy as np
from preprocessing import *
from plotting import *

try:
    # [OPTIONAL] Seaborn makes plots nicer
    import seaborn
except ImportError:
    pass

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
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(avg["micro"]))
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
