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
import sys  # For exiting main
try:
    # [OPTIONAL] Seaborn makes plots nicer
    import seaborn
except ImportError:
    pass

# =====================================================================
# We will calculate the P-R curve for each classifier
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score
def execute(clf, clf_name, X_train, X_test, y_train, y_test, with_ovr=False):
    clf.fit(X_train, y_train)

    # Print best estimator in case of GridSearchCV
    try:
        print("{}".format(clf.best_estimator_))
    except:
        pass

    pred = clf.predict(X_test)
    score = f1_score(y_test, pred, average='weighted')
    acc = accuracy_score(y_test, pred)

    # Generate the P-R curve
    try:
        y_prob = clf.decision_function(X_test)
    except AttributeError:
        # Handle BernoilliNB
        y_prob = clf.predict_proba(X_test)

    precision, recall, avg = get_per_class_pr_re_and_avg(y_test, y_prob) if with_ovr else (0.,0.,0.)

    # Include the score in the title
    print('{} (F1 score={:.3f}, Accuracy={:.4f})'.format(clf_name, score, acc))

    if with_ovr:
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
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(), y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score, average="micro")
    return precision, recall, average_precision

# =====================================================================

if __name__ == '__main__':
    # Import some classifiers to test
    from sklearn.svm import LinearSVC, NuSVC, SVC
    from sklearn.naive_bayes import BernoulliNB, MultinomialNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.model_selection import GridSearchCV

    url = r'C:\Users\vaibh\Source\Repos\DMTM_UIC\project_2_train\data 2_train.csv'

    #================= START: DEMO SETUP ===================================================
    ''' Remove this section when not using for demo '''

    test_url = r'C:\Users\vaibh\Source\Repos\DMTM_UIC\project_2_train\Data-2_test.csv'

    # Download the data set from URL
    print("Downloading training data from {}".format(url))
    frame_train = download_data(url)

    # Process data into feature and label arrays
    print("Processing {} training samples with {} attributes".format(len(frame_train.index), len(frame_train.columns)))
    X_train, y_train, fnames_trian = transform_features_and_labels(frame_train)

    # Download the data set from URL
    print("Downloading test data from {}".format(test_url))
    frame_test = download_data(test_url)

    # Process data into feature and label arrays
    print("Processing {} test samples with {} attributes".format(len(frame_test.index), len(frame_test.columns)))
    X_test, y_test, fnames_test = transform_features_and_labels(frame_test, skip_label=True)

    # Keep intersection of features in train and test (for dimension compatibility)
    i_keep_train = np.nonzero(np.in1d(fnames_trian, fnames_test))[0]
    i_keep_test = np.nonzero(np.in1d(fnames_test, fnames_trian))[0]
    X_train = X_train[:, i_keep_train]
    X_test = X_test[:, i_keep_test]

    print("Training classifier..")
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    score = f1_score(y_test, pred, average='weighted')
    acc = accuracy_score(y_test, pred)
    print('{} (F1 score={:.3f}, Accuracy={:.4f})'.format("LinearSVC", score, acc))

    print("Writing predictions to file..")
    import os
    result_file = "result.txt";
    try:
        os.remove(result_file)
    except OSError:
        pass
    file = open(result_file, "w") 
    frame_test_arr = np.array(frame_test)

    for i,p in enumerate(pred):
        file.write("{};;{}\n".format(frame_test_arr[i][0], p))
    file.close()

    sys.exit(0)

    #================= END: DEMO SETUP ===================================================

    # Download the data set from URL
    print("Downloading data from {}".format(url))
    frame = download_data(url)

    # Process data into feature and label arrays
    print("Processing {} samples with {} attributes".format(len(frame.index), len(frame.columns)))

    X, y, _ = transform_features_and_labels(frame, [-1,0,1])

    # Use 80% of the data for training, using CV; test against the rest
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Uncomment the following to test for using explicit 10-Fold CV and SVM (linear).
    # Read method description for details.
    #from extras import execute_using_cv
    #execute_using_cv(X, y)
    #sys.exit(0)

    # Evaluate multiple classifiers on the data
    print("Evaluating classifiers")
    classifiers = []
    classifiers.append([LinearSVC(), "LinearSVC"])
    classifiers.append([BernoulliNB(), "BernoulliNB"])
    classifiers.append([MultinomialNB(), "MultiNB"])
    classifiers.append([KNeighborsClassifier(n_neighbors=10), "kNN"])
    classifiers.append([AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm='SAMME.R'), "AdaBoost"])
    classifiers.append([RandomForestClassifier(n_estimators=100), "Random forest"])
    #classifiers.append([SVC(kernel='rbf', class_weight='balanced', decision_function_shape='ovo'), "Baseline ovo rbf SVM"])

    param_grid = {'C': np.logspace(-2, 1, 10),
                  #'kernel': ['rbf', 'linear', 'poly'],
                  #'gamma': [0.0005, 0.001, 0.005, 0.01, 0.1, 1.0],     # Use for testing rbf, poly
                 'degree': [1,2,3]}
    svm = GridSearchCV(SVC(kernel='linear', class_weight='balanced'), param_grid, cv=5)
    #classifiers.append([svm, "SVM with GridSearchCV (5-fold)"])

    results = []

    for clf in classifiers:
        results.append(execute(clf[0], clf[1], X_train, X_test, y_train, y_test))
        #results.append(execute(OneVsRestClassifier(clf[0]), clf[1], X_train, X_test, y_train, y_test, True))  # Use with label-binarized classes.

    # Display the results
    #print("Plotting the results")
    #for r in results:
    #    plot_avg_p_r_curves(r[0], r[1], r[2])
    #    plot_per_class_p_r_curves(r[0], r[1], r[2], [-1,0,1])   # Update classes