# [Lundberg2019] S. M. Lundberg et al. Explainable AI for Trees: From Local Explanations to Global Understanding. CoRR. 2019. http://arxiv.org/abs/1905.04610

"""
El código funciona bajo Python v3.7.0/v3.8.0 64-bits y ha sido testeado en Windows, Mac y Linux con las siguientes versiones de módulos:

numpy         ==> Windows: v1.19.3, Mac/Linux: v1.2.0
scikit-learn  ==> Windows: v0.23.2, Mac/Linux: v0.24.1
matplotlib    ==> Windows: v3.3.3
shap          ==> v0.38.1 - Info descarga: https://github.com/slundberg/shap
pandas        ==> Windows: v1.1.4
"""

import math
import copy
import itertools
import numpy as np
import pandas as pd
import coalitions_graph as cgraph

from sklearn.base import is_classifier  
from graph_plot_controller import GraphPlotController


# Controlador singleton de la vista
controller = GraphPlotController()


# Clase que representa un explicador del algoritmo TreeSHAP de [Lundberg2019]

class TreeShapExplainer: 

    # Constructor de la clase TreeShapExplainer
    
    def __init__(self, fid, enable_visual, dataset_name):
        self.cg = cgraph.CoalitionsGraph() 
        self.nfeatures_threshold = 5 # Límite de características del modelo para mostrar el grafo
        self.fid = fid
        self.enable_visual = enable_visual
        self.dataset_name = dataset_name


    # Función que genera una lista compuesta por todas las listas que pueden formarse reordenando los elementos de un cjto

    def __all_possible_orderings(self, feature_set):
        return [list(p) for p in itertools.permutations(feature_set, len(feature_set))]


    # Función que devuelve la predicción media (valor base) del modelo
    # E.g. nº de instancias en cada nodo hoja por el valor de salida obtenido entre el total de instancias
    # Ref: https://medium.com/analytics-vidhya/shap-part-3-tree-shap-3af9bcd7cd9b  

    def __calculate_base_value(self, X_train, tree, is_classifier):

            # Si se trata con un modelo clasificador hay que normalizar el número de instancias que cae en cada nodo
            values = self.__normalize_values(tree) if is_classifier else tree.value

            phi_zero = 0
            for i in range(tree.node_count):
                if tree.children_left[i] == tree.children_right[i]:
                    phi_zero += (tree.n_node_samples[i] * values[:][i])
                    
            return phi_zero / len(X_train) if is_classifier else (phi_zero / len(X_train))[0]


    # Función que obtiene la esperanza condicional de un conjunto S sobre un árbol aplicando el algoritmo (1) de [Lundberg2019] sobre una instancia
    # Nota: es f(x) en la ecuación (5) de [Lundberg2019]

    def __EXPVALUE(self, X, S, tree, is_classifier):

        # Se limpian etiquetas innecesarias y se normalizan los valores de los nodos si se trata con un modelo clasificador
        X = X.values
        v = self.__normalize_values(tree) if is_classifier else tree.value[:, 0, :]	

        return self.__G(X, S, 0, tree.children_left, tree.children_right, v, tree.feature, tree.threshold, tree.weighted_n_node_samples)


    # Algoritmo (1) de [Lundberg2019], que realiza el cálculo de la esperanza condicionada de S sobre una instancia X

    def __G(self, X, S, j, a, b, v, d, t, r):
        
        # Caso nodo hoja
        if b[j] == -1:
            return v[j]
        else:
            if d[j] in S:
                return self.__G(X, S, a[j], a, b, v, d, t, r) if X[0][d[j]] <= t[j] else self.__G(X, S, b[j], a, b, v, d, t, r)
            else:
                return (self.__G(X, S, a[j], a, b, v, d, t, r) * r[a[j]] + self.__G(X, S, b[j], a, b, v, d, t, r) * r[b[j]]) / r[j]


    # Función que por cada posible ordenación que toma, la separa en tantas listas como elementos, 
    # siendo las consecutivas acumulativas con la lista anterior
    # Ej: Para esta lista: [0,1,2] el resultado sería: [[0], [0,1], [0,1,2]], así con todas las posibles
    # Esta preparación facilita el cálculo de las diferencias de f(x) en la ecuación (5) de [Lundberg2019]

    def __prepare_orderings(self, all_orderings):

        acc_elements = []
        current_ordering = []
        total = []

        for ordering in all_orderings:
            for element in ordering:
                current_ordering.append(acc_elements + [element])
                acc_elements.append(element)

            acc_elements.clear()
            total += [current_ordering.copy()]
            current_ordering.clear()

        return total


    # Método para darle formato a la salida con la explicación final de los cálculos de valores SHAP

    def __print_explanation(self, X, dict, n_orderings, is_classifier):

        for i in dict:
            self.fid.write("\nΦ_" + str(X.columns[i]) + " = (")
            c = 0
            for j in dict[i]:
                c += 1
                if c != len(dict[i]):
                    if is_classifier: self.fid.write(str(j) + " + ")
                    else: self.fid.write(str(j[0]) + " + ")
                else:
                    if is_classifier: self.fid.write(str(j))
                    else: self.fid.write(str(j[0]))
                    
            if is_classifier:
                self.fid.write(") / " + str(n_orderings) + " = " + str(sum(dict[i]) / n_orderings))
            else:
                self.fid.write(") / " + str(n_orderings) + " = " + str((sum(dict[i]) / n_orderings)[0]))


    # Aplicación directa de la ecuación (5) de [Lundberg2019] sobre todas las características
    # Recibe una única instancia para generar los valores de SHAP de cada característica
    # y explicar la procedencia de su cálculo final

    def __shap_values_explainer(self, X, prepared_orderings, tree, base_value, is_classifier, G):
        
        # Inicialización de un diccionario que lleva los resultados de las diferencias 
        # para cada variable añadida sobre cada ordenación de todas las posibles en la ecuación (5)
        dict = {}
        for i in range(0, X.shape[1]): 
            dict[i] = []

        prev = 0
        i_ordering = 1
        self.fid.write("========= Calculating base value for the model...\n\n")
        self.fid.write("Base value Φ_0 = " + str(base_value) + " and corresponds to f(S) when S is empty\n\n")
        self.fid.write("Let f(x) be the conditional expectation function of the model's output\n")

        for ordering in prepared_orderings:
            self.fid.write("\n>>> Considering ordering #" + str(i_ordering) + " --> " + str([X.columns[[S][-1]] for S in ordering[-1]]) + "\n")

            prev = base_value
            for S in ordering:
                pr = self.__EXPVALUE(X, S, tree, is_classifier)
                p =  pr - prev  # f(x) con la característica añadida - f(x) anterior        

                if self.enable_visual == True and len(X.columns) < self.nfeatures_threshold:
                    self.cg.set_graph_attributes(G, [X.columns[s] for s in S], pr, prev, str(X.columns[S[-1]]))

                if is_classifier: self.fid.write("Φ_" + str(X.columns[S[-1]]) + "_" + str(i_ordering) + " = f(S U " + str(X.columns[S[-1]]) + ") - f(S) = " + str(pr) + " - " + str(prev) + " = " + str(p))
                else: self.fid.write("Φ_" + str(X.columns[S[-1]]) + "_" + str(i_ordering) + " = f(S U " + str(X.columns[S[-1]]) + ") - f(S) = " + str(pr[0]) + " - " + str(prev[0]) + " = " + str(p[0]))
                self.fid.write("\nS = " + str([X.columns[[i][-1]] for i in S]) + "\n")
                
                prev = pr
        
                dict[S[-1]] += [p]  # El sumatorio, pero llevando el de todas las características a la vez
            
            i_ordering += 1

        M_factorial = math.factorial(X.shape[1])    # M!
        self.fid.write("\n\n========= Summing up all feature values...\n\n")
        self.fid.write("M! = " + str(M_factorial) + "\n")
        self.__print_explanation(X, dict, M_factorial, is_classifier)
        
        shap_values_r = [sum(dict[i]) / M_factorial for i in dict]

        # Para cada categoría se recorre la matriz obteniendo los valores de su columna correspondiente
        shap_values_c = [[round(column[-j],5) for column in shap_values_r] for j in range(len(tree.value[0][0]), 0, -1)]

        if is_classifier:
            controller.set_shapley_values(shap_values_c)
            return shap_values_c
        else: 
            controller.set_shapley_values(shap_values_r)
            return shap_values_r


    # Función que normaliza el número de instancias que cae en cada nodo después de hacer un división
    # Ref: https://stackoverflow.com/questions/8904694/how-to-normalize-a-2-dimensional-numpy-array-in-python-less-verbose

    def __normalize_values(self, tree):
        filas_sumadas = tree.value[:, 0, :].sum(axis=1)
        return tree.value[:, 0, :] / filas_sumadas[:, np.newaxis]


    # Método que muestra a través de la salida los pasos seguidos para el cálculo de los valores SHAP
    # usando una instancia sobre un modelo de árbol entrenado previamente 

    def explain_the_explainer(self, X_train, X_test, tree_model):

        self.fid.write("========= Dataset name: " + self.dataset_name + "\n\n")
        G = []

        # Conjunto de características
        feature_set = list(range(0, len(X_train.columns)))

        # Obtención de todas las posibles ordenaciones de las características
        all_orderings = self.__all_possible_orderings(feature_set)
        prepared_orderings = self.__prepare_orderings(all_orderings)

        # Se crea el grafo de coalición
        if self.enable_visual == True and len(X_train.columns) < self.nfeatures_threshold: 
            G = self.cg.create_coalition_graph(self.__orderings_to_features(X_train,prepared_orderings), len(feature_set))

        # Cálculo del valor base
        base_value = self.__calculate_base_value(X_train, tree_model.tree_, is_classifier(tree_model))
        if self.enable_visual == True and len(X_train.columns) < self.nfeatures_threshold: 
            self.cg.set_graph_attributes(G, "{ }", base_value, 0, "{ }")
        
        # Cálculo y explicación de los valores SHAP
        shap_values = self.__shap_values_explainer(X_test, prepared_orderings, tree_model.tree_, base_value, is_classifier(tree_model), G)
        self.fid.write("\n\n\n============================== SHAP values from explainer ==============================\n")
        self.fid.write(str(shap_values))

        # Se dibuja el grafo de coalición
        if self.enable_visual == True and len(X_train.columns) < self.nfeatures_threshold: 
            self.cg.draw_coalition_graph(G)


    # Función que dada las ordenaciones de índices preparadas, las cambia a ordenaciones de características para el grafo

    def __orderings_to_features(self, X_train, prepared_orderings):
        copy_ = copy.deepcopy(prepared_orderings)
        j,k = 0,0
        for lst in copy_:
            k = 0
            for ordering in lst:
                copy_[j][k] = [X_train.columns[i] for i in ordering]    
                k += 1
            j += 1

        return copy_