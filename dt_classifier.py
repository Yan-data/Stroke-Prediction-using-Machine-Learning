from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, precision_score
# from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import GridSearchCV
import pickle
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.tree import plot_tree

RANDOM_STATE = 42

def read_from_pickle(filename):
    with open(filename, 'rb') as f:
        pickle_obj = pickle.load(f)
    return pickle_obj


def main():
    # load data from pickle files
    X_train = read_from_pickle("pickledata/X_train.pickle")
    y_train = read_from_pickle("pickledata/y_train.pickle")
    X_test = read_from_pickle("pickledata/X_test.pickle")
    y_test = read_from_pickle("pickledata/y_test.pickle")


    #X_train = X_train[:,[0,5,6]]
    #X_test = X_test[:,[0,5,6]]


    dec_clf = DecisionTreeClassifier(random_state=42, class_weight='balanced', criterion='log_loss', max_depth=2, max_features= 2)
    dec_clf.fit(X_train, y_train)
    print('feature importaince:',dec_clf.feature_importances_)
    print('n classes:', dec_clf.n_classes_)
    print('max feature :', dec_clf.max_features_)

    print(f'the most important feature x[{(dec_clf.feature_importances_).argmax()}]')
    plt.figure()
    plot_tree(dec_clf, filled=True)
    # plot_tree wurde importiert, wie gut, dass wir den nicht selbst schreiben müssen
    plt.show()

    print('depth', dec_clf.get_depth())
    print('leaves', dec_clf.get_n_leaves())

    y_pred = dec_clf.predict(X_train)
    #y_pred_test = dec_clf.predict(X_test)
    f11 = f1_score(y_train, y_pred)
    #f1 = f1_score(y_test, y_pred_test, average=None)
    #print("F1 SCORE TEST:", f1)
    precision = precision_score(y_train, y_pred)
    recall = recall_score(y_train, y_pred)
    print("train scores:", precision, recall, f11)


    # cross validation
    # cross validate scores for accuracy, recall and precision
    dec_cv_score_2 = cross_validate(dec_clf, X_train, y_train, cv=5, scoring=('precision', 'recall', 'f1'),return_train_score=True)
    #print(f"CV Test Score: {dec_cv_score_2}")
    print(f"CV Train Score: {np.average(dec_cv_score_2['train_precision']): .3f}, {np.average(dec_cv_score_2['train_recall']): .3f}, {np.average(dec_cv_score_2['train_f1']): .3f}")
    print(f"CV Test Score: {np.average(dec_cv_score_2['test_precision']): .3f}, {np.average(dec_cv_score_2['test_recall']): .3f}, {np.average(dec_cv_score_2['test_f1']): .3f}")
    #
    # #grid search
    #
    parameters = {"max_features": (2, 4, 6, 8),
                  "max_depth": (2, 4, 8, 16),
                  #'min_weight_fraction_leaf': (0, 0.01, 0.1, 0.2),
                  'criterion': ('gini', 'log_loss', 'entropy')

                  }
    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight='balanced'), parameters, scoring="recall", refit=True, cv=5, n_jobs=-1,
                               return_train_score=True)
    grid_search.fit(X_train, y_train)

    print("GridSearch best params:", grid_search.best_params_)
    print("GridSearch best score:", grid_search.best_score_)
    # print(f"AVG Accuracy Train Score: {grid_search.score(X_train, y_train): .3f}") #differs from knn_clf.score ???
    # # TODO: unclear why gridsearch.score differs from a knn_instance.score with best_params_




if __name__ == "__main__":
    main()
