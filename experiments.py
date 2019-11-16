from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import mutual_info_classif

from funcoes import *
from constantes import *

class Experiments():
    def __init__(self, dataDF):
        self.datasetDF = dataDF.rename(
                columns={'mean radius':'radius_mean', 'mean texture':'texture_mean','mean perimeter':'perimeter_mean',
                         'mean area':'area_mean', 'mean smoothness':'smoothness_mean',
                         'mean compactness':'compactness_mean',
                         'mean concavity':'concavity_mean', 'mean concave points':'concave points_mean',
                         'mean symmetry':'symmetry_mean', 'mean fractal dimension':'fractal_dimension_mean',
                         'radius error':'radius_se', 'texture error':'texture_se', 'perimeter error':'perimeter_se',
                         'area error':'area_se','smoothness error':'smoothness_se',
                         'compactness error':'compactness_se',
                         'concavity error':'concavity_se', 'concave points error':'concave points_se',
                         'symmetry error':'symmetry_se', 'fractal dimension error':'fractal_dimension_se',
                         'worst radius':'radius_worst', 'worst texture':'texture_worst',
                         'worst perimeter':'perimeter_worst',
                         'worst area':'area_worst', 'worst smoothness':'smoothness_worst',
                         'worst compactness':'compactness_worst', 'worst concavity':'concavity_worst',
                         'worst concave points':'concave points_worst', 'worst symmetry':'symmetry_worst',
                         'worst fractal dimension':'fractal_dimension_worst'})
        self.datasetDF.replace('B', 0, inplace=True)
        self.datasetDF.replace('M', 1, inplace=True)
        self.X, self.y = self.datasetDF.drop(CLASSE, axis=1), self.datasetDF[CLASSE]

    def decision_tree_images(self):
        createImageToDecisionTree(self.datasetDF, CLASSE) #sem remoção de atributos

        createImageToDecisionTree(self.datasetDF, CLASSE, 'concavity_mean')
        createImageToDecisionTree(self.datasetDF, CLASSE, 'concave points_mean')
        createImageToDecisionTree(self.datasetDF, CLASSE, 'concavity_worst')

        #todas apresentaram diferenças
        createImageToDecisionTree(self.datasetDF, CLASSE, 'concavity_worst')
        createImageToDecisionTree(self.datasetDF, CLASSE, 'concavity_mean')
        createImageToDecisionTree(self.datasetDF, CLASSE, 'compactness_worst')
        createImageToDecisionTree(self.datasetDF, CLASSE, 'concave points_worst')

        createImageToDecisionTree(self.datasetDF, CLASSE, 'area_worst')
        createImageToDecisionTree(self.datasetDF, CLASSE, 'radius_mean')
        createImageToDecisionTree(self.datasetDF, CLASSE, 'perimeter_mean')
        createImageToDecisionTree(self.datasetDF, CLASSE, 'area_mean')
        createImageToDecisionTree(self.datasetDF, CLASSE, 'radius_worst')
        createImageToDecisionTree(self.datasetDF, CLASSE, 'perimeter_worst')

        createImageToDecisionTree(self.datasetDF, CLASSE, 'perimeter_worst')
        createImageToDecisionTree(self.datasetDF, CLASSE, 'radius_mean')
        createImageToDecisionTree(self.datasetDF, CLASSE, 'perimeter_mean')
        createImageToDecisionTree(self.datasetDF, CLASSE, 'area_mean')
        createImageToDecisionTree(self.datasetDF, CLASSE, 'radius_worst')
        createImageToDecisionTree(self.datasetDF, CLASSE, 'area_worst')

        createImageToDecisionTree(self.datasetDF, CLASSE, 'area_se')
        createImageToDecisionTree(self.datasetDF, CLASSE, 'radius_se')
        createImageToDecisionTree(self.datasetDF, CLASSE, 'perimeter_se')

    def compare_grid_search_tree(self):
        self.X, self.y = self.datasetDF.drop(CLASSE, axis=1), self.datasetDF[CLASSE]
        print(" Com todos os atributos")
        self.gridsearch_tree()

        print("\nSem alguns atributos")
        dataframe = self.datasetDF
        df2 = dataframe.drop(columns=['concavity_mean','concave points_mean','area_mean','perimeter_mean','perimeter_se','perimeter_se'])
        self.X, self.y = df2.drop(CLASSE, axis=1), df2[CLASSE]
        self.gridsearch_tree()

    def gridsearch_tree(self):
        """
        Experiment which consists on applying gridsearch to find and evaluate the best decision tree estimator.
        The results are the parameters, the classification report and the confusion matrix, and they are printed on the console.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=0)

        parameters = {'min_samples_split': [0.0002, 0.25, 0.5], 'random_state': [1]}
        model = DecisionTreeClassifier()
        clf = GridSearchCV(model, parameters, cv=5)

        clf.fit(X_train, y_train)

        print(f"best parameters: \n{clf.best_params_}")

        y_pred = clf.best_estimator_.predict(X_test)

        print("\nclassification_report:\n")
        print(md_classification_report(y_test, y_pred))

        print("\nconfusion matrix:\n")
        print(md_confusion_matrix(y_test, y_pred))

    def gridsearch_svm(self):
        """
        Experiment which consists on applying gridsearch to find and evaluate the best SVM estimator.
        The results are the parameters, the classification report and the confusion matrix, and they are printed on the console.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=0)

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

    def gridsearch_svm_tree_knn(self):
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
            'DecisionTreeClassifier': {'min_samples_split': [0.0002, 0.25, 0.5], 'random_state': [1]},
            'KNeighborsClassifier': {'n_neighbors':[2**2,2**4,2**6, 2**8]}
        }
        helper = EstimatorSelectionHelper(models, params)
        helper.fit(self.X, self.y, scoring='f1', cv=KFold(n_splits=10, random_state=0))
        return helper.score_summary().sort_values('estimator')[['estimator', 'params', 'mean_fit_time',
                                                                'std_fit_time', 'mean_score_time',
                                                                'std_score_time',  'split0_test_score',
                                                                'split1_test_score','split2_test_score',
                                                                'split3_test_score', 'split4_test_score',
                                                                'split5_test_score', 'split6_test_score',
                                                                'split7_test_score','split8_test_score',
                                                                'split9_test_score', 'mean_test_score',
                                                                'std_test_score']]


    def mutual_entopy(self):
        mi = mutual_info_classif(self.X, self.y, discrete_features='auto', n_neighbors=3, copy=True, random_state=None)
        result = pd.DataFrame([mi], columns = self.X.columns).T.sort_values(by=0).T
    #     print(md_table(result))
        return result
