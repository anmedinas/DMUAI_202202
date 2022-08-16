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
