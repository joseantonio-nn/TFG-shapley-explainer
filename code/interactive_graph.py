""" 
Módulo construido y adaptado sobre el que se incluye en el siguiente link:
https://stackoverflow.com/questions/12894985/how-to-make-a-click-able-graph-by-networkx


El código funciona bajo Python v3.7.0/v3.8.0 64-bits y ha sido testeado en Windows, Mac y Linux con las siguientes versiones de módulos:

networkx      ==> Windows: v2.5.1 - pip install networkx
matplotlib    ==> Windows: v3.3.3
pylab         ==> Windows: v3.3.3 (es un módulo de matplotlib realmente)
"""

import networkx as nx
import matplotlib.pyplot as plt

from pylab import *
from matplotlib import rcParams
from graph_plot_controller import GraphPlotController


# Controlador singleton de la vista
controller = GraphPlotController()

# Clase que convierte un grafo construido con Networkx en interactivo
# Agradecimientos especiales al siguiente artículo http://www.scipy.org/Cookbook/Matplotlib/Interactive_Plotting

class AnnoteFinder:  
    """
    callback for matplotlib to visit a node (display an annotation) when points are clicked on.  The
    point which is closest to the click and within xtol and ytol is identified.
    """

    def __init__(self, xdata, ydata, annotes, pos, G, axis=None, xtol=None, ytol=None):
        rcParams['toolbar'] = 'None' 
        self.data = list(zip(xdata, ydata, annotes))
        if xtol is None: xtol = ((max(xdata) - min(xdata))/float(len(xdata)))/2
        if ytol is None: ytol = ((max(ydata) - min(ydata))/float(len(ydata)))/2
        self.xtol = xtol
        self.ytol = ytol
        if axis is None: axis = gca()
        self.axis= axis
        self.drawnAnnotations = {}
        self.links = []
        self.pos = pos
        self.G = G


    # Callback para cuando se haga click en el gráfico o se pulse una tecla

    def __call__(self, event):
        
        # Si se pulsa la tecla flecha derecha se avanza en la construcción del grafo
        if event.key == 'right':
            plt.close("Info")
            plt.close("Coalitions graph")
            controller.go_back(False)
        
        # Si se pulsa la tecla flecha izquierda se retrocede en la construcción del grafo
        elif event.key == 'left':
            plt.close("Info")
            plt.close("Coalitions graph")
            controller.go_back(True)
            controller.set_last_step(False)

        # Si se pulsa la tecla flecha arriba se completa la construcción del grafo
        elif event.key == 'up':
            plt.close("Info")
            plt.close("Coalitions graph")
            controller.set_last_step(True)
        
        # Si se pulsa la tecla 'escape' se cierra la ventana
        elif event.key == 'escape':
            plt.close("Info")
            plt.close("Coalitions graph")
            controller.set_quit(True)

        # Con la 'v' se pueden consultar los valores de Shapley finales
        elif event.key == 'v':
            plt.close("Shapley values")
            plt.figure("Shapley values",figsize=(8,2))
            plt.axis('off')
            
            plt.annotate(str(controller.get_shapley_values()), (0.5,0.5), va='center', xycoords='axes fraction', ha='center')
            plt.show(block=False)
            

        # Si se hace click en el grafo...
        elif event.inaxes:
            clickX = event.xdata
            clickY = event.ydata
    
            if self.axis is None or self.axis==event.inaxes:
                annotes = []
                smallest_x_dist = float('inf')
                smallest_y_dist = float('inf')

                # ... Se calculan las distancias con respecto a todos los nodos
                for x,y,a in self.data:
                    if abs(clickX-x)<=smallest_x_dist and abs(clickY-y)<=smallest_y_dist :
                        dx, dy = x - clickX, y - clickY
                        # Se toma el nodo más cercano si no se pasa de cierta distancia
                        if dx*dx+dy*dy < 0.000016: annotes.append((dx*dx+dy*dy, x, y, a))
                        smallest_x_dist=abs(clickX-x)
                        smallest_y_dist=abs(clickY-y)
    
                # Si hay nodos...
                if annotes:

                    # Se cierran las subventanas al abrir otras nuevas
                    plt.close("Info")
                    # Se selecciona el nodo más cercano al click
                    annotes.sort()
                    distance, x, y, annote = annotes[0]

                    # Se obtienen los atributos del nodo más cercano clickado y se le da formato
                    (S, pr, prev, new_feature) = self.position_to_node_attrs(x, y)

                    # Se crea la subventana y se muestran los atributos
                    plt.figure("Info",figsize=(8,2))
                    plt.axis('off')
                    info = "COALITION {" + str(S) + "}" if str(S) != "{ }" else "COALITION { }"
                    info += "\n\n The conditional expectation of the set is " + str(pr[0])
                    info += "\n\n The conditional expectation of the set without the addition was " + str(prev[0])
                    if new_feature != "{ }": info += "\n\n The marginal contribution of " + str(new_feature) + " in this set is " + str(pr[0]) + " - " + str(prev[0]) + " = " + str(pr[0]-prev[0])
                    plt.annotate(info, (0.5,0.5), va='center', xycoords='axes fraction', ha='center')
                    plt.show(block=False)

                    # Se crea un círculo encima del nodo seleccionado
                    self.drawAnnote(event.inaxes, x, y, annote)
        
                    
    # Función que dada una posición x e y devuelve los atributos del nodo al que corresponde en el gráfico

    def position_to_node_attrs(self, x, y):
        S = None
        for k,v in self.pos.items():
            if v[0] == x and v[1] == y:                            
                pr = self.G.nodes[k]["pr"]
                prev = self.G.nodes[k]["prev"]
                new_feature = self.G.nodes[k]["new_feature"]
                S = k
                break
        
        if not isinstance(pr[0], int) and not isinstance(pr[0], str) and len(pr[0]) == 1: pr = pr[0]
        else: pr[0] = np.array([round(n,3) for n in pr[0]])
            
        if not isinstance(prev[0], int) and not isinstance(prev[0], str) and len(prev[0]) == 1: prev = prev[0]
        elif not isinstance(prev[0], int): prev[0] = np.array([round(n,3) for n in prev[0]])

        return (S, pr, prev, "'"+new_feature[0]+"'")


    # Método que crea un círculo encima del nodo seleccionado para que se vea visualmente la selección

    def drawAnnote(self, axis, x, y, annote):
        
        # Se deseleccionan los nodos visualmente si se clica en otro distinto o en él de nuevo
        for k,v in self.drawnAnnotations.items():
            if self.drawnAnnotations[k]:
                for m in self.drawnAnnotations[k]:
                    m.set_visible(False)

        # Se dibuja el círculo blanco encima del nodo seleccionado
        t = axis.text(x, y, "",)
        m = axis.scatter([x], [y], marker='o', c='w', zorder=100, s=5*60, alpha=0.3)
        self.drawnAnnotations[(x, y)] = (t, m)

        self.axis.figure.canvas.draw()