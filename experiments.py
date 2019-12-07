
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from funcoes import *
from constantes import *
import warnings
warnings.filterwarnings('ignore')

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

    parameters = {'min_samples_split': [0.0002, 0.25, 0.5], 'random_state': [0]}
    model = DecisionTreeClassifier()
    clf = GridSearchCV(model, parameters, cv=5)

    clf.fit(X_train, y_train)

    print(f"best parameters: \n{clf.best_params_}")

    y_pred = clf.best_estimator_.predict(X_test)

    print("\nclassification_report:\n")
    print(md_classification_report(y_test, y_pred))

    print("\nconfusion matrix:\n")
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

    print("\nclassification_report:\n")
    print(md_classification_report(y_test, y_pred))

    print("\nconfusion matrix:\n")
    print(md_confusion_matrix(y_test, y_pred))

def gridsearch_svm_tree_knn(X, y):
    """
    Experiment which consists on applying gridsearch to find and evaluate the best of three estimator.
    The results are the parameters, the classification report and the confusion matrix, and they are printed on the console.
    """
    models = {
        'LinearSVC': LinearSVC(),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'KNeighborsClassifier': KNeighborsClassifier()
    }

    params = {
        'LinearSVC': {'C':[0.5, 2, 8, 32]},
        'DecisionTreeClassifier': {'min_samples_split': [0.0002, 0.25, 0.5], 'random_state': [0]},
        'KNeighborsClassifier': {'n_neighbors':[2**2,2**4,2**6, 2**8]}
    }
    helper = EstimatorSelectionHelper(models, params)
    helper.fit(X, y, scoring='f1', cv=KFold(n_splits=10, random_state=0))
#     print(md_table(helper.score_summary().sort_values('estimator')[['estimator', 'params', 'mean_fit_time',
#                                                                 'std_fit_time', 'mean_score_time',
#                                                                 'std_score_time',  'split0_test_score',
#                                                                 'split1_test_score','split2_test_score',
#                                                                 'split3_test_score', 'split4_test_score',
#                                                                 'split5_test_score', 'split6_test_score',
#                                                                 'split7_test_score','split8_test_score',
#                                                                 'split9_test_score', 'mean_test_score',
#                                                                 'std_test_score']]
# ))
    return helper


from sklearn.feature_selection import mutual_info_classif

def mutual_entopy(X, y):
    mi = mutual_info_classif(X, y, discrete_features='auto', n_neighbors=3, copy=True, random_state=None)
    result = pd.DataFrame([mi], columns = X.columns).T.sort_values(by=0).T
    print(md_table(result))

def meta_aprendizado(helper, X, y):
    print("Meta Aprendizado")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    y_pred = helper.election(X_test)

    print("\nclassification_report:\n")
    print(md_classification_report(y_test, y_pred))

    print("\nconfusion matrix:\n")
    print(md_confusion_matrix(y_test, y_pred))

if __name__ == "__main__":

    datasetDF = readDataset('id')
    datasetDF.replace('B', 0, inplace=True)
    datasetDF.replace('M', 1, inplace=True)
    X, y = datasetDF.drop(CLASSE, axis=1), datasetDF[CLASSE]

    # decision_tree_images(datasetDF, CLASSE)
    # print("Cálculo da entropia de cada atributo em relação à feature")
    # mutual_entopy(X, y)

    # print("Experimento gridsearch tree, comparação de atributos")
    # compare_grid_search_tree(datasetDF, CLASSE)

    # print("Experimento gridsearch svm")
    # gridsearch_svm(X, y)

    # print("Experimento grid search svm, decision tree e knn")
    helper = gridsearch_svm_tree_knn(X, y)
    meta_aprendizado(helper, X, y)
