# -*- coding: utf-8 -*-
import numpy as np
import warnings
import json
from resultado import Fold



class Experimento():
    def __init__(self,folds,ml_methods,num_folds_vc_treino=1):
        """
        folds: folds a serem usados no experimentos
        ml_methods: todos os métodos de aprendizado de máquina a serem testados. Geramente, são
                    apenas variações de parametros do mesmo método. Exemplo:
                    ml_methods =[DecisionTreeClassifier(min_samples_split=0.0002,random_state=1),
                                        DecisionTreeClassifier(min_samples_split=0.25,random_state=1),
                                        DecisionTreeClassifier(min_samples_split=0.5,random_state=1)]

        """
        self.folds = folds

        self.ml_methods = ml_methods
        self._resultados = None
        self.num_folds_vc_treino = num_folds_vc_treino
        
    def saveFolds(self, name):
        foldDict = {}
        for num_fold in range(len(self.folds)):
            df_treino  = self.folds[num_fold].df_treino
            df_teste  = self.folds[num_fold].df_data_to_predict
            foldDict[num_fold] = {"treino" : df_treino.index.values.tolist(), "teste" : df_teste.index.values.tolist()}        
        open(f'folds-{name}.txt', 'w').write(json.dumps(foldDict, indent=2))

    @property
    def macro_f1_avg(self):
        """
        Calcula a média do f1 dos resultados.
        """
        return np.average([r.macro_f1 for r in self.resultados])


    def obtem_validacao(self,fold):
        """
        A partir do fold, redivida o treino para fazer a validação.
        Use o atributo num_folds_vc_treino.

        Retorna um vetor de folds para que seja feito a validada

        Use o atributo num_folds_vc_treino para saber quantos folds será feito
        a validação cruzada no treino.
        Caso num_folds_vc_treino=1, então a validação possuirá o mesmo tamanho do teste e, o treino da validação será
        feito com o restante dos dados de treino.
        """        
        folds_validacao = []
        print(f'numFolds={self.num_folds_vc_treino}')
        #cria o fold validacao de acordo com o num_folds_vc_treino
        if(self.num_folds_vc_treino>1):
            #use o metodo gerar_k_folds para gerar a quantidade de folds desejada
            #utilize apenas o treino. A quantidade de folds será self.num_folds_vc_treino. A coluna da classe está no fold
            folds_validacao = Fold.gerar_k_folds(fold.df_treino,self.num_folds_vc_treino,fold.col_classe,seed=1)
        else:
            #caso seja =1, crie os dados de validação como uma amostra aleatoria do treino.
            #Faça essa amostra com o mesmo tamanho do teste
            #Use random_state = 1 para manter a mesma amostra
            print(f'treino {len(fold.df_treino)} predicct {len(fold.df_data_to_predict)}')
            df_validacao = fold.df_treino.sample(n=len(fold.df_data_to_predict), random_state=1)

            #crie o treino da validação: remova, do treino original, as instancias de validação
            df_treino_validacao = fold.df_treino.drop(df_validacao.index)

            #crie fold por meio do df_validacao e df_treino_validacao
            fold_validacao = Fold(df_treino_validacao,df_validacao,fold.col_classe)
            folds_validacao.append(fold_validacao)
        return folds_validacao

    def validacao_melhor_metodo(self,fold):
        """
        Usando o fold passado como parametro, faz a validação e obtem o melhor
        método dentre os self.ml_methods.

        Retorna o melhor método e, além disso, os resultados dos experimentos
        para cada método testado.

        O melhor método será obtido por meio do macro f1
        (atributo calculado macro_f1_avg da classe Experimento).
        """
        #1. extraia os folds de validação a partir do fold
        folds_validacao = self.obtem_validacao(fold)     

        #2. para cada ml_method, cria uma instancia de Experimento com ml_method e
        #.. verifica se este é o melhor resultado
        #em macro_f1
        best_result = 0
        best_method = None#este valor deve ser None mesmo :)
        arr_exp_validacao = []

        for ml_method in self.ml_methods:
            #instancie aqui o objeto da classe Experimento
            exp_validacao = exp_validacao = Experimento(folds_validacao,[ml_method],1);

            #Obtenha a melhor macro f1, use as variaveis best_result e best_method
            #para armazenar o melhor resultado até então
            if exp_validacao.macro_f1_avg > best_result:
                best_result = exp_validacao.macro_f1_avg
                best_method = exp_validacao
            
            #esse vetor é apenas para armazenarmos o resultado de cada validação
            arr_exp_validacao.append(exp_validacao)
            print(ml_method)
            print(exp_validacao.macro_f1_avg)
            print("---")
        #é retornado o melhor metodo e a lista de experimentos com seus resultados
        #caso seja necessario usa-los depois
        print("Best:")
        print(best_method.ml_methods)
        
        return arr_exp_validacao,best_method

    @property
    def resultados(self):
        """
        Retorna, para cada fold, o seu respectivo resultado
        """
        if self._resultados:
            return self._resultados
        #array de saida dos resultados
        self._resultados = []
        #Para cada fold, experimentos de validacao por método
        self.arr_validacao_por_fold = []

        ## Para cada fold i
        for i,fold in enumerate(self.folds):
            ##1. Obtem o melhor método, caso haja mais de um. Caso seja apenas um, use ele
            ##... Considere que sempre haverá mais de um método

            for ml_method in self.ml_methods:
            ##2. adiciona em resultados o resultado predito usando o melhor metodo
                self._resultados.append(fold.eval(ml_method))
        return self._resultados
