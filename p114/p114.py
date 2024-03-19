import sys
import argparse
import textwrap
import time
import matplotlib.pyplot as plt
import math
import numpy as np

from typing import List, Dict, Callable, Iterable, Tuple
#from p114 import *

#function at question IA 1
def matrix_multiplication(m_1: np.ndarray, m_2: np.ndarray)-> np.ndarray:
    # shape = (filas, columas)
    m_1_shape, m_2_shape = m_1.shape, m_2.shape
    if m_1_shape[1] != m_2_shape[0]:
        return None

    m_final = np.zeros(shape=(m_1_shape[0], m_2_shape[1]))
    # Filas de m_1
    for i in range(m_1_shape[0]):
        # Columnas de m_2
        for j in range(m_2_shape[1]):
            # Columnas de m_1 y filas de m_2 a la vez
            for k in range(m_1_shape[1]):
                m_final[i][j] += m_1[i][k] * m_2[k][j]

    return m_final


#function at question IB 1

#busqueda binaria
def rec_bb(t: List, f: int, l: int, key: int)-> int:
    #f: first
    #l: last
    if l >= f:
        m = (f + l) // 2
        if t[m] == key:
            return m
        elif t[m] > key:
            return rec_bb(t, f, m -1, key)
        else:
            return rec_bb(t, m+1, l, key)
    else:
        return None

#busqueda binaria lineal
#function at question IB 2
def bb(t: List, f: int, l: int, key: int) -> int:
    while f <= l:
        mid = (f + l) // 2
        if key == t[mid]:
            return mid
        elif key < t[mid]:
            l = mid-1
        else:
            f = mid+1


"""Ajusta linealmente los valores de la funcion func_2_fit a
los tiempos en timings.
Esto es, calculamos valores a, b para que la funcion a*f(dim) + b
se ajuste a los tiempos medidos.
"""
def fit_func_2_times(timings: np.ndarray, func_2_fit: Callable):
    if len(timings.shape) == 1:
        timings = timings.reshape(-1, 1)
    values = func_2_fit(timings[ :, 0]).reshape(-1, 1)
   
    #normalizar timings
    times = timings[ : , 1] / timings[0, 1]
    
    #ajustar a los valores en times un modelo lineal sobre los valores en values
    lr_m = LinearRegression()
    lr_m.fit(values, times)
    return lr_m.predict(values)

def func_2_fit(n):
    return  n**3


#function at question II A1
def min_heapify(h: np.ndarray, i: int):
    h_len = len(h) - 1
    if i < 0 or i > h_len:
        return None

    while 2*i+1 <= h_len:
        n_i = i
        # Compara padre con hijo izquierdo
        if h[i] > h[2*i+1]:
            n_i = 2*i+1
        # Compara padre o hijo izquierdo con hijo derecho
        if 2*i+2 <= h_len and h[i] > h[2*i+2] and h[2*i+2] < h[n_i]:
            n_i = 2*i+2
        # Si algun hijo es menor que el padre, se cambian
        if n_i > i:
            h[i], h[n_i] = h[n_i], h[i]
            i = n_i
        else:
            break

    return h

#function at question II A2
def insert_min_heap(h: np.ndarray, k: int)-> np.ndarray:
    if h == None:
        h = []

    h += [k]
    j = len(h) - 1

    while j >= 1 and h[(j-1) // 2] > h[j]:
        h[(j-1) // 2], h[j] = h[j], h[(j-1) // 2]
        j = (j-1) // 2

#function at question II A3
def create_min_heap(h: np.ndarray):
    j =h[((len(h)-1)-1) // 2]

    while j > -1:
        min_heapify(h, j)
        j -=1
    return h


#function at question II B1
#COLAS DE PRIORIDAD
#pq: priority queue
def pq_ini():
    pq = []
    return pq

#function at question II B2
def pq_insert(h: np.ndarray, k: int)-> np.ndarray:
    h += [k]
    j = len(h) - 1
    while j >= 1 and h[(j-1) // 2] > h[j]:
        h[(j-1) // 2], h[j] = h[j], h[(j-1) // 2]
        j = (j-1) // 2 #hijo
    return h

#function at question II B3
def pq_remove(h: np.ndarray)->Tuple[int, np.ndarray]:
    if len(h) == 0:
        return (h[0], h)
    
    e = h[0]
    h[0] = h[-1]
    h.pop()
    min_heapify(h, 0)
    return (e, h)

#function at question IIs C1

def select_min_heap(h: np.ndarray, k: int)-> int:
    h = [-i for i in h]
    l = h[k:]
    h = h[:k]

    for i in l:
        if h[0] < 1:
            h[0] = i
            min_heapify(h, 0)
    return -h[0]

def max_heapify(h: np.ndarray, i: int) -> np.ndarray:
    """Comprueba que los hijos son menores del padre que ocupa la posicion i.
    
    Modifica el heap de tal forma que el elemento de la posicion i, sus hijos sean menores que él.
    
    Args: 
    
        h (np.ndarray): un heap.
        
        i (int): la posicion del elemento el cual se quiere hacer heapify. 
        
    """

    left = 2 * i + 1
    right = 2 * i + 2

    while left < len(h):

        left = 2 * i + 1
        right = 2 * i + 2

        if left < len(h) and h[left] > h[i]:
            largest = left
        else:
            largest = i
        if right < len(h) and h[right] > h[largest]:
            largest = right
        if largest != i:
            h[i], h[largest] = h[largest], h[i]
            i = largest
        else: 
            return
            
    
def create_max_heap(h: np.ndarray) -> np.ndarray:
    """Crea un max heap.
    
    Crea un max heap sobre el array que le es pasado por argumento.
    
    Args: 
    
        h (np.ndarray): un array de Numpy.
        
    """
    for i in range(len(h) // 2 - 1, -1, -1):
        max_heapify(h, i)


def select_max_heap(h: np.ndarray, k: int) -> int:
    """Retorna el valor que ocuparia la posicion k en un array ordenado.
    
    Args: 
    
        h (np.ndarray): el min heap.
        
        k (int): la posicion que ocuparia
        
    Returns: 
    
        j (int): el elemento que ocuparia la posicion k en un array ordenado
        
    """
    h_aux = np.array([])
    h_aux = list(map(lambda n: n, h)) 

    first_k_elem = h_aux[:k]
    create_max_heap(first_k_elem)

    for i in h_aux[k:]: 
        if i < first_k_elem[0]:  
            first_k_elem[0]=i
            max_heapify(first_k_elem, 0)

    return first_k_elem[0]
    

    
def main(t_size: int):
    """Examen de practicas 2022
    """
    t = np.random.permutation(t_size)
    
    for k in range(1, t_size+1):
        val = select_max_heap(t, k)
        print('pos', k, '\tval', val)
        