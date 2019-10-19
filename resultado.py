# -*- coding: utf-8 -*-
from sklearn.exceptions import UndefinedMetricWarning
import numpy as np
import warnings
import json
class Resultado():
    def __init__(self, y, predict_y):
        """
        y: Vetor numpy (np.array) em que, para cada instancia i, y[i] é a classe alvo da mesma
        predict_y: Vetor numpy (np.array) que representa a predição y[i] para a instancia i

        Tanto y quando predict_y devem assumir valores numéricos
        """
        self.y = y
        self.predict_y = predict_y
        self._mat_confusao = None
        self._precisao = None
        self._revocacao = None

    @property
    def mat_confusao(self):
        """
        Retorna a matriz de confusão.
        """
        #caso a matriz de confusao já esteja calculada, retorna-la
        if self._mat_confusao  is not None:
            return self._mat_confusao

        #instancia a matriz de confusao como uma matriz de zeros
        #A matriz de confusão terá o tamanho como o máximo entre os valores de self.y e self.predict_y
        max_class_val = max([self.y.max(),self.predict_y.max()])    
        self._mat_confusao = np.zeros((max_class_val+1,max_class_val+1))



        #incrementa os valores da matriz baseada nas listas self.y e self.predict_y
        for i,classe_real in enumerate(self.y):
            self._mat_confusao[classe_real][self.predict_y[i]] += 1

        #print("Predict y: "+str(self.predict_y))
        #print("y: "+str(self.y))
        #print("Matriz de confusao final :"+str(self._mat_confusao))
        return self._mat_confusao

    @property
    def precisao(self):
        """
        Precisão por classe
        """
        if self._precisao is not None:
            return self._precisao

        #inicialize com um vetor de zero usando np.zeros
        self._precisao = np.zeros(len(self.mat_confusao))

        #para cada classe, armazene em self._precisao[classe] o valor relativo à precisão
        #dessa classe
        for classe in range(len(self.mat_confusao)):
            #obtnha todos os elementos que foram previstos com essa classe
            num_previstos_classe = 0
            for classe_real in range(len(self.mat_confusao)):
                num_previstos_classe += self.mat_confusao[classe_real][classe]

            #precisao: numero de elementos previstos corretamente/total de previstos com essa classe
            #calcule a precisão para a classe
            if num_previstos_classe!=0:
                self._precisao[classe] =  self.mat_confusao[classe][classe]/num_previstos_classe
            else:
                self._precisao[classe] = 0
                warnings.warn("Não há elementos previstos para a classe "+str(classe)+" precisão foi definida como zero.", UndefinedMetricWarning)
        return self._precisao
    @property
    def revocacao(self):
        if self._revocacao is not None:
            return self._revocacao

        self._revocacao = np.zeros(len(self.mat_confusao))
        for classe in range(len(self.mat_confusao)):
            #por meio da matriz, obtem todos os elementos que são dessa classe
            num_classe = 0
            num_elementos_classe = 0
            for classe_prevista in range(len(self.mat_confusao)):
                num_elementos_classe += self.mat_confusao[classe][classe_prevista]

            #revocacao: numero de elementos previstos corretamente/total de elementos dessa classe
            if num_elementos_classe!=0:
                self._revocacao[classe] =  self.mat_confusao[classe][classe]/num_elementos_classe
            else:
                self._revocacao[classe] = 0
                warnings.warn("Não há elementos da classe "+str(classe)+" revocação foi definida como zero.", UndefinedMetricWarning)
        return self._revocacao

    @property
    def f1_por_classe(self):
        """
        retorna um vetor em que, para cada classe, retorna o seu f1
        """
        f1 = np.zeros(len(self.mat_confusao))
        for classe in range(len(self.mat_confusao)):
            if(self.precisao[classe]+self.revocacao[classe] == 0):
                f1[classe] = 0
            else:
                f1[classe] = 2*(self.precisao[classe]*self.revocacao[classe])/(self.precisao[classe]+self.revocacao[classe])
        return f1

    @property
    def macro_f1(self):
        return np.average(self.f1_por_classe)

    @property
    def acuracia(self):
        #quantidade de elementos previstos corretamente
        num_previstos_corretamente = 0
        for classe in range(len(self.mat_confusao)):
            num_previstos_corretamente += self.mat_confusao[classe][classe]

        return num_previstos_corretamente/len(self.y)
class Fold():
    def __init__(self,df_treino,df_data_to_predict,col_classe):
        self.df_treino = df_treino
        self.df_data_to_predict = df_data_to_predict
        self.col_classe = col_classe


    def eval(self,ml_method):
        #a partir de self.df_treino, separe as features da classe
        x_treino = self.df_treino.drop(self.col_classe,axis=1)
        y_treino = self.df_treino[self.col_classe]



        #crie o modelo
        model = ml_method.fit(x_treino,y_treino)
        #separe as features a serem previstas e a classe
        x_to_predict = self.df_data_to_predict.drop(self.col_classe,axis=1)
        y_to_predict = self.df_data_to_predict[self.col_classe]

        #Impressao do x e y
        #print("X_treino: "+str(x_treino))
        #print("y_treino: "+str(y_treino))
        #print("X_to_predict: "+str(x_to_predict))
        #print("y_to_predict: "+str(y_to_predict))

        #retorne o resultado
        return Resultado(y_to_predict,model.predict(x_to_predict))

    @staticmethod
    def gerar_k_folds(df,val_k,col_classe,seed=1):
        """
        Retorna um vetor arr_folds com todos os k folds criados a partir do DataFrame df.

        df: DataFrame com os dados a serem usados
        k: Número de folds a serem gerados
        col_classe: coluna que representa a classe
        seed: seed para a amostra aleatória
        """
        #1. especifique o número de instancias
        #Aqui, você deve dividir o número de instancias len(df) pelo número de folds.
        #Qual é o melhor: arredondar sempre para cima (teto) ou para baixo (piso)? Escolha a melhor opção
        #O operador // faz o floor entre dois numeros, para o teto, seria math.ceil.
        #substitua o "None" abaixo
        num_instances_per_partition = len(df) // val_k 
        arr_folds = []

        #1. Embaralhe os dados: o método sample, ao usar 100% dos dados, embaralha os dados.
        #Use a seed passada como parametro.
        df_dados = df.sample(frac=1, random_state = seed )
        
        #Impressão dos ids dos dados (para testes)
        #print("Dados: "+str(df.index.values))

        #para cada fold num_fold:
        
        for num_fold in range(val_k):
            #2. especifique o inicio e fim da partição que representará o teste.
            #..Caso seja o ultimo, o fim será o tamanho do vetor.
            #..Use num_instances_per_partition e num_fold para deliminar o inicio e fim do teste
            #..caso haja dúvidas, veja figura na especificação
            ini_fold_teste = num_fold * num_instances_per_partition
            fim_fold_teste = len(df) if num_fold == (val_k - 1) else (ini_fold_teste + num_instances_per_partition)
            
            #3. obtenha o fold de teste por meio do
            df_teste = df_dados[ini_fold_teste:fim_fold_teste]

            #4. Crie o treino, removendo o teste dos dados originais (df).
            df_treino = df_dados.drop(df_teste.index)

            #5. Crie o fold (objeto da classe Fold) e adicione no vetor
            fold = Fold(df_treino,df_teste,col_classe)
            arr_folds.append(fold)


        """
        for num_fold in range(val_k):
            df_treino  = arr_folds[num_fold].df_treino
            df_teste  = arr_folds[num_fold].df_data_to_predict
            print("Fold #{num} instancias no treino: {qtd_treino} teste: {qtd_teste}"
                                                        .format(num=num_fold,
                                                        qtd_treino=len(df_treino.index),
                                                        qtd_teste=len(df_teste.index)))
            print("Fold #{num} idx treino: {treino}".format(num=num_fold,treino=df_treino.index.values))
            print("Fold #{num} idx teste: {teste}".format(num=num_fold,teste=df_teste.index.values))
        """
        return arr_folds
