def execute_using_cv(X, y, use_ovr=False):
    '''
    Stand-alone function to executes and plot evaluation metrics using explicit (averaged) 10-Fold CV and SVM.
    'use_ovr' for multiclass data.
    Ref: http://scikit-learn.org/stable/auto_examples/exercises/plot_cv_digits.html#sphx-glr-auto-examples-exercises-plot-cv-digits-py
    '''
    print(__doc__)

    import numpy as np
    from sklearn.model_selection import cross_val_score
    from sklearn import datasets, svm

    #digits = datasets.load_digits()
    #X = digits.data
    #y = digits.target
    svc = svm.SVC(kernel='linear')

    if use_ovr:
        from sklearn.multiclass import OneVsRestClassifier
        svc = OneVsRestClassifier(svc)

    C_s = np.logspace(-2, 1, 10)

    scores = list()
    scores_std = list()
    print("Running 10-Fold CV on Linear SVM, with varying C..")
    for C in C_s:
        svc.C = C
        this_scores = cross_val_score(svc, X, y, cv=10, n_jobs=4)
        mean_score = np.mean(this_scores)
        scores.append(mean_score)
        scores_std.append(np.std(this_scores))
        print("Score with C={} is {}".format(C, mean_score))

    # Do the plotting
    import matplotlib.pyplot as plt
    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.semilogx(C_s, scores)
    plt.semilogx(C_s, np.array(scores) + np.array(scores_std), 'b--')
    plt.semilogx(C_s, np.array(scores) - np.array(scores_std), 'b--')
    locs, labels = plt.yticks()
    plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
    plt.ylabel('CV score')
    plt.xlabel('Parameter C')
    plt.ylim(0, 1.1)
    plt.show()
