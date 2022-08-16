# -*- coding: utf-8 -*-

'''
@author: Andres C. Medina
Aprendizaje No Supervisado: Metodo Particion KMeans
'''

'''Modulos Elementales'''
from sklearn import datasets
import numpy as np
import pandas as pd

'''Modulos de Visualizacion'''
from matplotlib import pyplot as plt
from plotnine import *
import seaborn as sns

'''Modulo Clustering'''
from sklearn.cluster import KMeans

'''Otros Modulos'''
import warnings 
import sys

'''Configuraciones'''
warnings.simplefilter("ignore")
sns.set_style("dark")

'''Versiones Utilizadas'''
print("Python version:",sys.version)
print("Numpy version:",np.__version__)
print("Pandas version:",pd.__version__)
print("Seaborn version:",sns.__version__) 

'''Carga Set de Datos'''
iris = datasets.load_iris()
print(iris.keys(),"\n")

'''Contiene los datos''' 
print(iris.data)

'''Contiene el target numérico'''
print(iris.target)

'''Contiene el target como una marca o flag'''
print(iris.target_names)

'''Contiene la descripción del conjunto de datos'''
print(iris.DESCR)

'''Contiene las características o variables'''
print(iris.feature_names)

'''Contiene el path de donde se almacena el conjunto de datos'''
print(iris.filename)

'''Convierto a DataFrame el archivo iris data'''
tempDF = pd.DataFrame(iris.data, columns = iris.feature_names)

'''Muestro los cuatro primeros registros de iris data'''
print(tempDF.head(4))

#TODO Agregar Diagrama dispersion

'''Invocando de Cluster, la función Kmeans'''
km = KMeans()
print(km) 

'''Entrenamiento KMeans'''
km = km.fit(tempDF) 

'''Muestra los centros de los  clusters construidos'''
print(km.cluster_centers_)

'''Incercia o Within Cluster Distance'''
print(km.inertia_)

'''Etiquetas'''
print(km.labels_)

'''Iteraciones del algoritmo'''
print(km.n_iter_)

# TODO agregar visualizacion conglomerado.

'''Heuristica para encontrar K'''
sse = [] 
for k in range(1, 15): 
    # Creando el modelo, entrenandolo y obteniendo el valor del within cluster distance
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(iris.data)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(8,6))
plt.plot(range(1,15), sse, marker='o',linestyle='--')
plt.axvline(x=3, color='r', linestyle='--')
plt.title('KMeans',fontsize=15)
plt.xlabel('Numero de Clusters',fontsize=13)
plt.ylabel('WCSS', fontsize=13)
plt.show()





