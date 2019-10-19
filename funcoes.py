import pandas as pd
import json
from sklearn.tree import DecisionTreeClassifier
from arvoreDeDecisao import *

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
