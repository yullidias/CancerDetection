from sklearn.tree import export_graphviz
from IPython.display import Image  
import pydotplus
from sklearn.externals.six import StringIO  


def plotArvoreDeDecisao(arvoreDeDecisao, atributosDF, classeDF, nome_arquivo_saida):
    # Creates dot file named tree.dot
    sio = StringIO()
    export_graphviz(
                arvoreDeDecisao,
                out_file =  sio,            
                special_characters=True,
                feature_names = list(atributosDF.columns),
                class_names = [str(classe) for classe in set(classeDF)],
                filled = True,
                rounded = True,
                impurity = True,
                node_ids = True, # mostra o id de cada nó
                proportion = True,
                precision = True
    )

    graph = pydotplus.graph_from_dot_data(sio.getvalue())  
    graph.write_png(nome_arquivo_saida)
    Image(graph.create_png())
