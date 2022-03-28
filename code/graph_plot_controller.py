"""
El código funciona bajo Python v3.7.0/v3.8.0 64-bits y ha sido testeado en Windows, Mac y Linux
"""


# Clase controlador singleton para manejar la vista cómodamente

class GraphPlotController:

    __instance = None

    # Constructor del controlador

    def __init__(self):
        self.q = False
        self.lastStep = False
        self.backwards = False
        self.shapley_values = []


    # A partir de la segunda vez que se pida la instancia se devuelve la misma

    def __new__(cls):
        if GraphPlotController.__instance is None:
            GraphPlotController.__instance = object.__new__(cls)
        return GraphPlotController.__instance

    
    # Función que retorna un booleano indicando si se quiere salir

    def quit(self):
        return self.q


    # Función que retorna un booleano indicando si el grafo está completamente dibujado

    def is_last_step(self):
        return self.lastStep
    
    def is_backwards(self):
        return self.backwards

    def get_shapley_values(self):
        return self.shapley_values


    # Setters de las variables anteriores

    def set_quit(self, quit):
        self.q = quit

    def set_last_step(self, lastStep):
        self.lastStep = lastStep

    def go_back(self, backw):
        self.backwards = backw

    def set_shapley_values(self,shap):
        self.shapley_values = shap