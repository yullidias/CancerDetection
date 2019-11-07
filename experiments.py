
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from funcoes import *
from constantes import *

def decision_tree_images(datasetDF, target):
    createImageToDecisionTree(datasetDF, target) #sem remoção de atributos

    createImageToDecisionTree(datasetDF, target, 'concavity_mean')
    createImageToDecisionTree(datasetDF, target, 'concave points_mean')
    createImageToDecisionTree(datasetDF, target, 'concavity_worst')

    #todas apresentaram diferenças
    createImageToDecisionTree(datasetDF, target, 'concavity_worst')
    createImageToDecisionTree(datasetDF, target, 'concavity_mean')
    createImageToDecisionTree(datasetDF, target, 'compactness_worst')
    createImageToDecisionTree(datasetDF, target, 'concave points_worst')

    createImageToDecisionTree(datasetDF, target, 'area_worst')
    createImageToDecisionTree(datasetDF, target, 'radius_mean')
    createImageToDecisionTree(datasetDF, target, 'perimeter_mean')
    createImageToDecisionTree(datasetDF, target, 'area_mean')
    createImageToDecisionTree(datasetDF, target, 'radius_worst')
    createImageToDecisionTree(datasetDF, target, 'perimeter_worst')

    createImageToDecisionTree(datasetDF, target, 'perimeter_worst')
    createImageToDecisionTree(datasetDF, target, 'radius_mean')
    createImageToDecisionTree(datasetDF, target, 'perimeter_mean')
    createImageToDecisionTree(datasetDF, target, 'area_mean')
    createImageToDecisionTree(datasetDF, target, 'radius_worst')
    createImageToDecisionTree(datasetDF, target, 'area_worst')

    createImageToDecisionTree(datasetDF, target, 'area_se')
    createImageToDecisionTree(datasetDF, target, 'radius_se')
    createImageToDecisionTree(datasetDF, target, 'perimeter_se')

def compare_grid_search_tree(dataframe, target):
    X, y = datasetDF.drop(target, axis=1), datasetDF[target]
    print(" Com todos os atributos")
    gridsearch_tree(X, y)

    print("\nSem alguns atributos")
    df2 = dataframe.drop(columns=['concavity_mean','concave points_mean','area_mean','perimeter_mean','perimeter_se','perimeter_se'])
    X, y = df2.drop(target, axis=1), df2[target]
    gridsearch_tree(X, y)

def gridsearch_tree(X, y):
    """
    Experiment which consists on applying gridsearch to find and evaluate the best decision tree estimator.
    The results are the parameters, the classification report and the confusion matrix, and they are printed on the console.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 10, 100, 1000]}
    model = svm.SVC(gamma="scale")
    clf = GridSearchCV(model, parameters, cv=5)

    clf.fit(X_train, y_train)

    print(f"best parameters: \n{clf.best_params_}")

    y_pred = clf.best_estimator_.predict(X_test)

    print("\nclassification_report:")
    print(md_classification_report(y_test, y_pred))
 
    print("\nconfusion matrix:")
    print(md_confusion_matrix(y_test, y_pred))

def gridsearch_svm(X, y):
    """
    Experiment which consists on applying gridsearch to find and evaluate the best SVM estimator.
    The results are the parameters, the classification report and the confusion matrix, and they are printed on the console.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 10, 100, 1000]}
    model = svm.SVC(gamma="scale")
    clf = GridSearchCV(model, parameters, cv=5)

    clf.fit(X_train, y_train)

    print(f"best parameters: \n{clf.best_params_}")

    y_pred = clf.best_estimator_.predict(X_test)

    print("\nclassification_report:")
    print(md_classification_report(y_test, y_pred))
 
    print("\nconfusion matrix:")
    print(md_confusion_matrix(y_test, y_pred))

def gridsearch_svm_tree_knn(X, y):
    """
    Experiment which consists on applying gridsearch to find and evaluate the best of three estimator.
    The results are the parameters, the classification report and the confusion matrix, and they are printed on the console.
    """
    models = { 
        'SVC': SVC(),
        'LinearSVC': LinearSVC(),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'KNeighborsClassifier': KNeighborsClassifier()
    }

    params = { 
        'SVC': {'C':[0.5, 2, 8], 'gamma':[0.5, 2, 8]},
        'LinearSVC': {'C':[0.5, 2, 8, 32]},
        'DecisionTreeClassifier': {'min_samples_split': [0.0002, 0.25, 0.5], 'random_state': [1]},
        'KNeighborsClassifier': {'n_neighbors':[2**2,2**4,2**6, 2**8]}
    }
    helper = EstimatorSelectionHelper(models, params)
    helper.fit(X, y, scoring='f1', cv=KFold(n_splits=10, random_state=0))
    return helper.score_summary()

if __name__ == "__main__":

    datasetDF = readDataset('id')
    datasetDF.replace('B', 0, inplace=True)
    datasetDF.replace('M', 1, inplace=True)
    X, y = datasetDF.drop(CLASSE, axis=1), datasetDF[CLASSE]

    # decision_tree_images(datasetDF, CLASSE)
    
    print("Experimento gridsearch tree, comparação de atributos")
    compare_grid_search_tree(datasetDF, CLASSE)

    print("Experimento gridsearch svm")
    #gridsearch_svm(X, y)

    print("Experimento grid search svm, decision tree e knn")
    #gridsearch_svm_tree_knn(X, y)

