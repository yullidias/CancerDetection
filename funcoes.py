import pandas as pd
import json
from sklearn.tree import DecisionTreeClassifier
#from arvoreDeDecisao import *

def readDataset(index):
    return pd \
            .read_csv(open('pathToDataset.txt', 'r') \
            .read().split("\n")[0], index_col=index) \
            .drop(axis=1, labels='Unnamed: 32')

def readJson(path):
    with open(path, 'r') as f:
        return json.loads(f.read())

def createImageToDecisionTree(cancerDeMamaDF, nome_classe, atributosParaRemover=None,criterio='entropy'):
    if atributosParaRemover != None:
        cancerDeMamaDF = cancerDeMamaDF.drop(axis=1, labels=atributosParaRemover)

    atributos = list(cancerDeMamaDF.columns)
    atributos.remove(nome_classe)

    ml_methods_params = [DecisionTreeClassifier(criterion=criterio,min_samples_split=0.0002,random_state=1)]#,
                         #DecisionTreeClassifier(criterion=criterio,min_samples_split=0.25,random_state=1),
                         #DecisionTreeClassifier(criterion=criterio,min_samples_split=0.5,random_state=1)]

    folds = readJson('folds-tree.txt')
    for arvore in ml_methods_params:
        dataset = cancerDeMamaDF.loc[folds["1"]['treino']] #foi escolhido um fold por simplificação
        arvore.fit(dataset.drop(axis=1, labels=nome_classe), dataset[nome_classe])
        plotArvoreDeDecisao(arvore, cancerDeMamaDF[atributos], cancerDeMamaDF[nome_classe],
                            f'img/arvore_{arvore.min_samples_split}_removido-{atributosParaRemover}.png')


from tabulate import tabulate
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.model_selection import KFold

TARGETS = ["B", "M"]

def md_table(dataframe):
    """
    Returns a markdown styled table given a dataframe.
    """
    return tabulate(dataframe, tablefmt="pipe", headers="keys")

def md_classification_report(y_true, y_pred, targets=TARGETS):
    """
    Returns the classification report table, markdown styled, given y_true and y_pred.
    based on https://gist.github.com/wassname/f3cbdc14f379ba9ec2acfafe5c1db592
    """
    report = pd.DataFrame(classification_report(y_true, y_pred, target_names=targets, output_dict=True)).T
    report[["precision","recall","f1-score"]] = report[["precision","recall","f1-score"]].apply(lambda x: round(x,2))
    report[["support"]]= report[["support"]].apply(lambda x: x.astype(np.int))
    report[["support"]]= report[["support"]].apply(lambda x: x.astype(np.int))
    report.loc['accuracy', 'support'] = report.loc['weighted avg', 'support']
    return md_table(report)

def md_confusion_matrix(y_true, y_pred, targets=TARGETS):
    """
    Returns the confusion matrix table, markdown styled, given y_true and y_pred.
    """
    columns = [f"{t}_pred" for t in targets]
    index = [f"{t}_true" for t in targets]

    matrix = pd.DataFrame(confusion_matrix(y_true, y_pred), columns=columns, index=index)
    return md_table(matrix)

def kfold_split(X, y, k):
    """
    Split the dataset represented by the features X and target y into k folds.
    parameters:
        X (dataframe): the feature matrix
        y (dataframe): the target vector
        k (int): the number of folds to split.
    response:
        result (list o tuples (X_train, X_test, y_train, y_test)):
            list of training and testing folds
    """
    result = []
    for train_index, test_index in KFold(n_splits=k).split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        result.append((X_train, X_test, y_train, y_test))
    return result

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV

class EstimatorSelectionHelper:
    """
    from https://github.com/davidsbatista/machine-learning-notebooks/blob/master/hyperparameter-across-models.ipynb
    adapted from: http://www.codiply.com/blog/hyperparameter-grid-search-across-multiple-models-in-scikit-learn/
    """

    def __init__(self, models, params):
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, **grid_kwargs):
        for key in self.keys:
            print('Running GridSearchCV for %s.' % key)
            model = self.models[key]
            params = self.params[key]
            grid_search = GridSearchCV(model, params, **grid_kwargs)
            grid_search.fit(X, y)
            self.grid_searches[key] = grid_search
        print('Done.')

    def score_summary(self, sort_by='mean_test_score'):
        frames = []
        for name, grid_search in self.grid_searches.items():
            frame = pd.DataFrame(grid_search.cv_results_)
            frame = frame.filter(regex='^(?!.*param_).*$')
            frame['estimator'] = len(frame)*[name]
            frames.append(frame)
        df = pd.concat(frames)

        df = df.sort_values([sort_by], ascending=False)
        df = df.reset_index()
        df = df.drop(['rank_test_score', 'index'], 1)

        new_columns_order = ['estimator'] + [col for col in df if col not in ['estimator']]
        return df[new_columns_order]

    def get_bests(self):
        bests = []
        for model in self.grid_searches:
            bests.append(self.grid_searches[model].best_estimator_)
        print(bests)
        return bests

    def election(self,X):
        y_pred_for_model = []
        result = []
        for best in self.get_bests():
            predicted = best.predict(X)
            y_pred_for_model.append(predicted)
        for j in range(len(y_pred_for_model[0])):
            sum_predicted = 0
            for i in range(len(y_pred_for_model)):
                sum_predicted += y_pred_for_model[i][j]
            result.append(round(sum_predicted/float(len(y_pred_for_model))))
        return result
