import io # utf-8 encoding para output
import shap
import pandas as pd
import explaining_treeshap_explainer as ete

from sklearn.tree import DecisionTreeClassifier


# Carga del conjunto Iris
X_train,y = shap.datasets.iris()
X_train_copy = X_train.copy()
        
for i in range(1, len(X_train.columns) + 1):
    X_train_copy.rename(columns={X_train.columns[i-1] : str(i)}, inplace=True)

# Entrenamiento de un árbol de decisión
tree_model = DecisionTreeClassifier(max_depth=2, min_samples_leaf=1, min_samples_split=2, random_state = 100)
tree_model.fit(X_train_copy, y)

# Nueva instancia para predecir
X_test = pd.DataFrame({X_train.columns[0]: [1], X_train.columns[1]: [1], X_train.columns[2]: [1], X_train.columns[3] : [1]})

explainer = shap.TreeExplainer(tree_model)

shap_values = explainer.shap_values(X_test)

X_test = pd.DataFrame({'1': [1], '2': [1], '3': [1], '4' : [1]})

filename = "iris.txt"
fid = io.open(filename, "w", encoding="utf-8")

tse = ete.TreeShapExplainer(fid, enable_visual=True, dataset_name="iris")
tse.explain_the_explainer(X_train_copy, X_test, tree_model)

fid.write("\n\n============================== SHAP values from shap module ==============================\n" + str(shap_values))
fid.close()