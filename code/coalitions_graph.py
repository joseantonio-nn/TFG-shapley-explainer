"""
El código funciona bajo Python v3.7.0/v3.8.0 64-bits y ha sido testeado en Windows, Mac y Linux con las siguientes versiones de módulos:

numpy          ==> Windows: v1.19.3, Mac/Linux: v1.2.0
matplotlib     ==> Windows: v3.3.3
networkx       ==> Windows: v2.5.1 - pip install networkx
"""

import re
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
# Soporte para MAC
matplotlib.use('TkAgg')
#matplotlib.use('Qt4Agg')
# Para leyenda
from matplotlib.lines import Line2D

from interactive_graph import AnnoteFinder,connect
from graph_plot_controller import GraphPlotController


# Controlador singleton de la vista
controller = GraphPlotController()


# Clase que se encarga de la construcción y manejo de un grafo de coalición

class CoalitionsGraph:

    # Constructor de la clase CoalitionsGraph

    def __init__(self):
        self.G = nx.DiGraph()
        self.color_map = []
        self.prepared_orderings = []
        self.feature_set_len = 0
        self.edges = None
        self.nodes = None


    # Función que construye el grafo de coalición G y lo devuelve

    def create_coalition_graph(self, prepared_orderings, feature_set_len):
        
        self.feature_set_len = feature_set_len
        self.prepared_orderings = prepared_orderings

        # El primer nodo se pinta de color rojo
        self.G.add_node("{ }", pr=[], prev=[], new_feature=[])    

        # Se crean los nodos 
        for i in range(len(prepared_orderings)):
            for j in prepared_orderings[i]:
                self.G.add_node(self.list_to_formatted_string(j), pr=[], prev=[], new_feature=[])

        # Se unen los nodos con los ejes correspondientes
        for i in range(len(prepared_orderings)):

            aux = prepared_orderings[i][0]
            fst = True
            for j in prepared_orderings[i]:
                self.G.add_edge("{ }" if fst else self.list_to_formatted_string(aux), self.list_to_formatted_string(j), nf=str(j[-1]))
                aux = j
                fst = False
        
        self.edges = list(self.G.edges())
        self.nodes = list(self.G.nodes())
        return self.G


    # Método que establece la contribución marginal del nodo del grafo asociado a un subconjunto de una ordenación 

    def set_graph_attributes(self, G, subordering, pr, prev, new_feature):

        if subordering == "{ }":
            key = subordering
        else:
            key = self.list_to_formatted_string(subordering)

        G.nodes[key]["pr"].append(pr)
        G.nodes[key]["prev"].append(prev)
        G.nodes[key]["new_feature"].append(new_feature)

        return key


    # Método que dibuja el grafo de coalición G en una imagen
    # La imagen se crea el directorio desde donde se invoca al código

    def draw_coalition_graph(self, G):
        
        i = 0
        pos = nx.planar_layout(G, scale=0.1) 
        while i < len(self.nodes):

            # Si se quiere salir, se rompe el bucle
            if controller.quit():
                break

            plt.close("Coalitions graph")

            # Se copia el grafo original para no perderlo en cada iteración
            Gaux = G.copy()

            # Se puede ir hacia atrás 
            if controller.is_backwards() and i > 0:
                i -= 2
                if i < 0: i = 0
                Gaux.remove_edges_from(self.edges[i:])
                Gaux.remove_nodes_from([e[1] for e in self.edges[i:]])
                i += 1

            # Si se ha avanzado hasta la vista final, el iterador es el valor máximo para que acabe el bucle
            elif not controller.is_last_step():
                
                # Se borran los ejes y nodos desde el final hasta el i para simular su construcción
                Gaux.remove_edges_from(self.edges[i:])
                Gaux.remove_nodes_from([e[1] for e in self.edges[i:]])
                i += 1
    
            else:
                i = len(self.nodes)
            
            # Se establece el layout de los nodos
            #pos = nx.planar_layout(Gaux, scale=0.1) 
            
            # Se asocian las posiciones de los nodos para el gráfico interactivo
            x, y, annotes = [], [], []
            for key in pos:
                d = pos[key]
                annotes.append(key)
                x.append(d[0])
                y.append(d[1])

            # Formato de la ventana principal
            fig = plt.figure("Coalitions graph", figsize=(20,50))
            ax = fig.add_subplot(111)
            ax.set_title('Explainer of TreeSHAP explainer')

            # Leyenda
            legend_elements = [
                    Line2D([0], [0], marker='o', color='w', label='Right arrow to go one step forwards', markerfacecolor='black', markersize=7),
                    Line2D([0], [0], marker='o', color='w', label='Left arrow to go one step backwards', markerfacecolor='black', markersize=7),
                    Line2D([0], [0], marker='o', color='w', label='Up arrow to build the complete graph', markerfacecolor='black', markersize=7),
                    Line2D([0], [0], marker='o', color='w', label='V to show final Shapley values', markerfacecolor='black', markersize=7),
                    Line2D([0], [0], marker='o', color='w', label='Esc to exit', markerfacecolor='black', markersize=7)]

            ax.legend(handles=legend_elements, loc='upper right')

            

            # Se generan los colores del grafo
            color_map = self.generate_color_map(Gaux)

            # Formato del grafo
            edge_labels = nx.get_edge_attributes(Gaux,'nf')
            posAux = dict(list(pos.items()))
            
            for n in G.nodes:
                if n not in Gaux.nodes: 
                    del posAux[n]
            nx.draw(Gaux, pos=posAux, with_labels=True, node_size=[5*60 for i in posAux], node_shape='o', node_color=color_map)
            nx.draw_networkx_edge_labels(Gaux, posAux, edge_labels = edge_labels)

            # Callback para hacerlo interactivo y detectar teclado
            af = AnnoteFinder(x, y, annotes, pos, Gaux)
            connect('button_press_event', af)
            connect('key_press_event', af)
            
            # Se dibuja el grafo
            plt.get_current_fig_manager().window.wm_geometry("+-5+0")
            if self.feature_set_len == 4:
                plt.xlim([-0.12,0.12])
                plt.ylim([-0.015,0.013])
            else:
                plt.xlim([-0.12,0.12])
                plt.ylim([-0.03,0.031])

            # Si se llega al último nodo se avisa
            if i >= len(self.nodes):
                plt.figure("Info",figsize=(3,1))
                plt.axis('off')
                plt.annotate("Graph has been completely built.", (0.5,0.5), va='center', xycoords='axes fraction', ha='center')
                plt.show(block=False)
            plt.show()



    # Función que transforma una lista en un 'string' formateado para mejor visualización

    def list_to_formatted_string(self, alist):
        s = ", ".join(["{}"] * len(alist)).format(*alist)
        return s.format(*alist)


    # Función que genera el mapa de color para el grafo

    def generate_color_map(self, Gaux):
        fst = True
        color_map = []
        color_map.append((0.749,0.616,0.961)) # Lila 
        
        for n in Gaux:
            if fst == True:
                fst = False
                continue 
            
            # Se detectan los nodos hoja para darle otro color
            lenActual = len(re.findall('[1-9]+', n)) 
            if self.feature_set_len != lenActual:
                color_map.append((0.435,0.733,0.827)) # Azul claro    
            else:
                color_map.append((0.6,0.898,0.6)) # Verde claro

        return color_map
