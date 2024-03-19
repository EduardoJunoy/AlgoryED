import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from typing import List, Tuple, Dict, Callable, Iterable

"""Practica 1
    """

"""matrix_multiplication(m_1, m_2) - Cambio: División en lugar de multiplicación.
"""
import numpy as np

def matrix_division(m_1, m_2):
    """Recibe dos matrices Numpy y devuelve otra matriz Numpy con su división"""
    n_rows, n_cols = m_1.shape[0], m_1.shape[1]
    m_division = np.zeros((n_rows, n_cols))

    for i in range(n_rows):
        for j in range(n_cols):
            if m_2[i, j] != 0:
                m_division[i, j] = m_1[i, j] / m_2[i, j]
            else:
                m_division[i, j] = np.inf
    
    return m_division
"""rec_bb(t: List, f: int, l: int, key: int) - Cambio: Modificación del algoritmo de vuelta atrás.
"""
def rec_bb(t, f, l, key):
    """Implementa el algoritmo de vuelta atrás con modificaciones"""
    if f == l:
        return f

    mid = (f + l) // 2

    if t[mid] == key:
        return mid

    if t[f] < t[mid]:
        if t[f] <= key <= t[mid]:
            return rec_bb(t, f, mid - 1, key)
        return rec_bb(t, mid + 1, l, key)
    else:
        if t[mid] <= key <= t[l]:
            return rec_bb(t, mid + 1, l, key)
        return rec_bb(t, f, mid - 1, key)
"""bb(t: List, f: int, l: int, key: int) - Cambio: Modificación del algoritmo de ramificación y acotación.
"""
def bb(t, f, l, key):
    """Implementa el algoritmo de ramificación y acotación con modificaciones"""
    while f <= l:
        mid = (f + l) // 2

        if t[mid] == key:
            return mid

        if t[f] < t[mid]:
            if t[f] <= key <= t[mid]:
                l = mid - 1
            else:
                f = mid + 1
        else:
            if t[mid] <= key <= t[l]:
                f = mid + 1
            else:
                l = mid - 1

    return -1

"""
min_heapify(h: np.ndarray, i: int) - Cambio: Modificación de la función de min_heapify.
python
"""
import numpy as np

def min_heapify(h, i):
    """Reorganiza el heap min en la posición i"""
    n = len(h)
    smallest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and h[left] < h[smallest]:
        smallest = left

    if right < n and h[right] < h[smallest]:
        smallest = right

    if smallest != i:
        h[i], h[smallest] = h[smallest], h[i]
        min_heapify(h, smallest)
"""insert_min_heap(h: np.ndarray, k: int) - Cambio: Modificación de la función de inserción en el heap min.
python
"""
import numpy as np

def insert_min_heap(h, k):
    """Inserta un elemento en el heap min"""
    h = np.append(h, k)
    i = len(h) - 1

    while i > 0 and h[(i - 1) // 2] > h[i]:
        h[i], h[(i - 1) // 2] = h[(i - 1) // 2], h[i]
        i = (i - 1) // 2

"""matrix_multiplication(m_1, m_2) - Cambio: Resta en lugar de multiplicación.
python"""

import numpy as np

def matrix_subtraction(m_1, m_2):
    """Recibe dos matrices Numpy y devuelve otra matriz Numpy con su resta"""
    n_rows, n_cols = m_1.shape[0], m_1.shape[1]
    m_subtraction = np.zeros((n_rows, n_cols))

    for i in range(n_rows):
        for j in range(n_cols):
            m_subtraction[i, j] = m_1[i, j] - m_2[i, j]
    
    return m_subtraction
"""rec_bb(t: List, f: int, l: int, key: int) - Cambio: Cambio del valor del pivote en el algoritmo de vuelta atrás.
python"""

def rec_bb(t, f, l, key):
    """Implementa el algoritmo de vuelta atrás con cambio del valor del pivote"""
    if f == l:
        return f

    mid = (f + l) // 2

    # Cambio del pivote de 5 a 3
    pivot = 3

    if t[mid] == key:
        return mid

    if t[f] < t[mid]:
        if t[f] <= key <= t[mid]:
            return rec_bb(t, f, mid - 1, key)
        return rec_bb(t, mid + 1, l, key)
    else:
        if t[mid] <= key <= t[l]:
            return rec_bb(t, mid + 1, l, key)
        return rec_bb(t, f, mid - 1, key)
"""bb(t: List, f: int, l: int, key: int) - Cambio: Cambio del valor del pivote en el algoritmo de ramificación y acotación.
"""
def bb(t, f, l, key):
    """Implementa el algoritmo de ramificación y acotación con cambio del valor del pivote"""
    while f <= l:
        mid = (f + l) // 2

        # Cambio del pivote de 5 a 3
        pivot = 3

        if t[mid] == key:
            return mid

        if t[f] < t[mid]:
            if t[f] <= key <= t[mid]:
                l = mid - 1
            else:
                f = mid + 1
        else:
            if t[mid] <= key <= t[l]:
                f = mid + 1
            else:
                l = mid - 1

    return -1
"""min_heapify(h: np.ndarray, i: int) - Cambio: Alteración del criterio de orden en el heap min.
"""
import numpy as np

def min_heapify(h, i):
    """Reorganiza el heap min en la posición i con un criterio de orden alterado"""
    n = len(h)
    smallest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and h[left] > h[smallest]:  # Alteración del criterio de orden
        smallest = left

    if right < n and h[right] > h[smallest]:  # Alteración del criterio de orden
        smallest = right

    if smallest != i:
        h[i], h[smallest] = h[smallest], h[i]
        min_heapify(h, smallest)
"""QSelect(arr: List, k: int) - Cambio: Uso del pivote 3 en lugar del pivote 5 en el algoritmo de selección rápida.
"""
def QSelect(arr, k):
    """Implementa el algoritmo de selección rápida con el uso del pivote 3"""
    if len(arr) == 1:
        return arr[0]

    pivot = arr[3]  # Uso del pivote 3 en lugar del pivote 5

    lesser = [x for x in arr if x < pivot]
    equal = [x for x in arr if x == pivot]
    greater = [x for x in arr if x > pivot]

    if k <= len(lesser):
        return QSelect(lesser, k)
    elif k <= len(lesser) + len(equal):
        return equal[0]
    else:
        return QSelect(greater, k - len(lesser) - len(equal))
    
    
"""matrix_multiplication(m_1, m_2) - Cambio: División en lugar de multiplicación.
"""
import numpy as np

def matrix_division(m_1, m_2):
    """Recibe dos matrices Numpy y devuelve otra matriz Numpy con su división"""
    n_rows, n_cols = m_1.shape[0], m_1.shape[1]
    m_division = np.zeros((n_rows, n_cols))

    for i in range(n_rows):
        for j in range(n_cols):
            if m_2[i, j] != 0:
                m_division[i, j] = m_1[i, j] / m_2[i, j]
    
    return m_division
"""rec_bb(t: List, f: int, l: int, key: int) - Cambio: Búsqueda binaria iterativa en lugar de recursiva.
"""
def rec_bb_iterative(t, f, l, key):
    """Implementa el algoritmo de búsqueda binaria iterativa"""
    while f <= l:
        mid = (f + l) // 2

        if t[mid] == key:
            return mid

        if t[mid] < key:
            f = mid + 1
        else:
            l = mid - 1
    
    return -1
"""bb(t: List, f: int, l: int, key: int) - Cambio: Orden inverso en el algoritmo de ramificación y acotación.
"""
def bb_reverse_order(t, f, l, key):
    """Implementa el algoritmo de ramificación y acotación con orden inverso"""
    while f <= l:
        mid = (f + l) // 2

        if t[mid] == key:
            return mid

        if t[f] > t[mid]:  # Orden inverso
            if t[f] >= key >= t[mid]:  # Orden inverso
                l = mid - 1
            else:
                f = mid + 1
        else:
            if t[mid] >= key >= t[l]:  # Orden inverso
                f = mid + 1
            else:
                l = mid - 1

    return -1
"""min_heapify(h: np.ndarray, i: int) - Cambio: Uso de heap max en lugar de heap min.
"""
import numpy as np

def max_heapify(h, i):
    """Reorganiza el heap max en la posición i"""
    n = len(h)
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and h[left] > h[largest]:
        largest = left

    if right < n and h[right] > h[largest]:
        largest = right

    if largest != i:
        h[i], h[largest] = h[largest], h[i]
        max_heapify(h, largest)
"""QSelect(arr: List, k: int) - Cambio: Orden inverso en el algoritmo de selección rápida.
"""
def QSelect_reverse_order(arr, k):
    """Implementa el algoritmo de selección rápida con orden inverso"""
    if len(arr) == 1:
        return arr[0]

    pivot = arr[4]  # Pivote 5

    lesser = [x for x in arr if x > pivot]  # Orden inverso
    equal = [x for x in arr if x == pivot]
    greater = [x for x in arr if x < pivot]  # Orden inverso

    if k <= len(lesser):
        return QSelect_reverse_order(lesser, k)
    elif k <= len(lesser) + len(equal):
        return equal[0]
    else:
        return QSelect_reverse_order(greater, k - len(lesser) - len(equal))
    
"""matrix_multiplication(m_1, m_2) - Cambio: Suma en lugar de multiplicación.
"""
import numpy as np

def matrix_sum(m_1, m_2):
    """Recibe dos matrices Numpy y devuelve otra matriz Numpy con su suma"""
    n_rows, n_cols = m_1.shape[0], m_1.shape[1]
    m_sum = np.zeros((n_rows, n_cols))

    for i in range(n_rows):
        for j in range(n_cols):
            m_sum[i, j] = m_1[i, j] + m_2[i, j]
    
    return m_sum
"""rec_bb(t: List, f: int, l: int, key: int) - Cambio: Búsqueda ternaria en lugar de búsqueda binaria.
"""
def rec_ternary_search(t, f, l, key):
    """Implementa el algoritmo de búsqueda ternaria recursiva"""
    if f <= l:
        mid1 = f + (l - f) // 3
        mid2 = l - (l - f) // 3

        if t[mid1] == key:
            return mid1
        elif t[mid2] == key:
            return mid2
        elif key < t[mid1]:
            return rec_ternary_search(t, f, mid1 - 1, key)
        elif key > t[mid2]:
            return rec_ternary_search(t, mid2 + 1, l, key)
        else:
            return rec_ternary_search(t, mid1 + 1, mid2 - 1, key)
    
    return -1
"""bb(t: List, f: int, l: int, key: int) - Cambio: Orden aleatorio en el algoritmo de ramificación y acotación.
"""
import random

def bb_random_order(t, f, l, key):
    """Implementa el algoritmo de ramificación y acotación con orden aleatorio"""
    while f <= l:
        mid = (f + l) // 2

        if t[mid] == key:
            return mid

        pivot = random.choice([t[f], t[mid], t[l]])  # Selección aleatoria del pivote

        if t[f] > pivot:
            if t[f] >= key >= pivot:
                l = mid - 1
            else:
                f = mid + 1
        else:
            if pivot >= key >= t[l]:
                f = mid + 1
            else:
                l = mid - 1

    return -1
"""min_heapify(h: np.ndarray, i: int) - Cambio: Uso de heap máximo en lugar de heap mínimo con valores negativos.
"""
import numpy as np

def max_heapify_negative(h, i):
    """Reorganiza el heap máximo en la posición i con valores negativos"""
    n = len(h)
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and h[left] < h[largest]:  # Uso de heap máximo con valores negativos
        largest = left

    if right < n and h[right] < h[largest]:  # Uso de heap máximo con valores negativos
        largest = right

    if largest != i:
        h[i], h[largest] = h[largest], h[i]
        max_heapify_negative(h, largest)
"""insert_min_heap(h: np.ndarray, k: int) - Cambio: Inserción en heap máximo con valores negativos.
"""
import numpy as np

def insert_max_heap_negative(h, k):
    """Inserta un elemento en el heap máximo con valores negativos"""
    h.append(-float('inf'))  # Valores negativos
    i = len(h) - 1

    while i > 0 and h[(i - 1) // 2] < k:  # Uso de heap máximo con valores negativos
        h[i] = h[(i - 1) // 2]
        i = (i - 1) // 2

    h[i] = k

"""matrix_multiplication(m_1, m_2) - Cambio: Resta en lugar de multiplicación.
"""
import numpy as np

def matrix_subtraction(m_1, m_2):
    """Recibe dos matrices Numpy y devuelve otra matriz Numpy con su resta"""
    n_rows, n_cols = m_1.shape[0], m_1.shape[1]
    m_subtraction = np.zeros((n_rows, n_cols))

    for i in range(n_rows):
        for j in range(n_cols):
            m_subtraction[i, j] = m_1[i, j] - m_2[i, j]
    
    return m_subtraction
"""rec_bb(t: List, f: int, l: int, key: int) - Cambio: Búsqueda exponencial en lugar de búsqueda binaria.
"""
def rec_exponential_search(t, f, l, key):
    """Implementa el algoritmo de búsqueda exponencial recursiva"""
    if f > l:
        return -1

    if t[f] == key:
        return f

    if f == l or t[l] == key:
        return l

    mid = f + (l - f) // 2

    if t[mid] == key:
        return mid

    if key < t[mid]:
        return rec_exponential_search(t, f, mid - 1, key)
    else:
        return rec_exponential_search(t, mid + 1, l, key)
"""bb(t: List, f: int, l: int, key: int) - Cambio: Búsqueda cuádruple en lugar de búsqueda binaria.
"""
def bb_quadruple_search(t, f, l, key):
    """Implementa el algoritmo de búsqueda cuádruple"""
    while f <= l:
        mid1 = f + (l - f) // 4
        mid2 = f + (l - f) // 2
        mid3 = l - (l - f) // 4

        if t[mid1] == key:
            return mid1
        elif t[mid2] == key:
            return mid2
        elif t[mid3] == key:
            return mid3
        elif key < t[mid1]:
            l = mid1 - 1
        elif key < t[mid2]:
            f = mid1 + 1
            l = mid2 - 1
        elif key < t[mid3]:
            f = mid2 + 1
            l = mid3 - 1
        else:
            f = mid3 + 1

    return -1
"""min_heapify(h: np.ndarray, i: int) - Cambio: Uso de heap mínimo con valores absolutos.
"""
import numpy as np

def min_heapify_abs(h, i):
    """Reorganiza el heap mínimo en la posición i con valores absolutos"""
    n = len(h)
    smallest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and abs(h[left]) < abs(h[smallest]):
        smallest = left

    if right < n and abs(h[right]) < abs(h[smallest]):
        smallest = right

    if smallest != i:
        h[i], h[smallest] = h[smallest], h[i]
        min_heapify_abs(h, smallest)
"""insert_min_heap(h: np.ndarray, k: int) - Cambio: Inserción en heap mínimo con valores opuestos.
"""
import numpy as np

def insert_min_heap_opposite(h, k):
    """Inserta un elemento en el heap mínimo con valores opuestos"""
    h.append(-k)  # Valores opuestos

    i = len(h) - 1

    while i > 0 and h[(i - 1) // 2] > h[i]:
        h[i], h[(i - 1) // 2] = h[(i - 1) // 2], h[i]
        i = (i - 1) // 2

"""##############################################################################################################################################################"""
"""##############################################################################################################################################################"""
"""##############################################################################################################################################################"""
"""##############################################################################################################################################################"""


"""Practica2"""

"""init_cd(n: int) -> np.ndarray: Cambio: Inicialización de un conjunto disjunto con representantes aleatorios.
"""

def init_cd_random(n: int) -> np.ndarray:
    """Inicializa un conjunto disjunto con representantes aleatorios"""
    return np.random.randint(0, n, size=n)
"""union(rep_1: int, rep_2: int, p_cd: np.ndarray) -> int: Cambio: Unión de dos nodos de un conjunto disjunto junto con la profundidad del conjunto obtenido.
"""

def union_with_depth(rep_1: int, rep_2: int, p_cd: np.ndarray) -> int:
    """Une dos nodos de un conjunto disjunto y devuelve el representante del conjunto obtenido junto con la profundidad"""
    p_cd[rep_1] = rep_2
    return rep_2, p_cd[rep_2] + 1



"""find(ind: int, p_cd: np.ndarray) -> int: Cambio: Encuentra la raíz del nodo ind y devuelve la profundidad del conjunto si el nodo ya es padre.
"""

import numpy as np

def find_with_depth(ind: int, p_cd: np.ndarray) -> int:
    """Encuentra la raíz del nodo `ind` y devuelve el representante o la profundidad si el nodo ya es padre"""
    if p_cd[ind] == ind:
        return ind, 0
    
    rep, depth = find_with_depth(p_cd[ind], p_cd)
    p_cd[ind] = rep
    return rep, depth + 1

"""cd_2_dict(p_cd: np.ndarray) -> Dict: Cambio: Crea un diccionario a partir de un conjunto disjunto, pero incluye el tamaño de cada componente conexa.
"""
from collections import defaultdict

def cd_2_dict_with_size(p_cd: np.ndarray) -> Dict:
    """Crea un diccionario a partir de un conjunto disjunto, incluyendo el tamaño de cada componente conexa"""
    component_dict = defaultdict(list)
    for i in range(len(p_cd)):
        root = find(i, p_cd)
        component_dict[root].append(i)
    
    component_dict_with_size = {}
    for root, component in component_dict.items():
        component_dict_with_size[root] = {
            'representative': root,
            'size': len(component),
            'nodes': component
        }
    
    return component_dict_with_size
"""greedy_tsp(dist_m: np.ndarray, node_ini=0) -> List: Cambio: Genera un circuito codicioso a partir de una matriz y un nodo de inicio, pero permite una lista de nodos prohibidos.
"""
def greedy_tsp_with_forbidden_nodes(dist_m: np.ndarray, node_ini=0, forbidden_nodes=[]) -> List:
    """Genera un circuito codicioso a partir de una matriz y un nodo de inicio, permitiendo nodos prohibidos"""
    n = len(dist_m)
    circuit = [node_ini]
    current_node = node_ini

    while len(circuit) < n:
        min_dist = float('inf')
        next_node = None

        for i in range(n):
            if i not in circuit and i not in forbidden_nodes:
                if dist_m[current_node][i] < min_dist:
                    min_dist = dist_m[current_node][i]
                    next_node = i

        circuit.append(next_node)
        current_node = next_node

    return circuit


"""repeated_greedy_tsp(dist_m: np.ndarray) -> List: Cambio: Aplica el algoritmo greedy_tsp múltiples veces, pero con un límite de iteraciones.
"""
def repeated_greedy_tsp_with_limit(dist_m: np.ndarray, limit=100) -> List:
    """Aplica el algoritmo greedy_tsp múltiples veces con un límite de iteraciones"""
    best_circuit = None
    best_length = float('inf')

    n = len(dist_m)
    for _ in range(limit):
        circuit = greedy_tsp(dist_m)
        length = len_circuit(circuit, dist_m)

        if length < best_length:
            best_circuit = circuit
            best_length = length

    return best_circuit
"""exhaustive_tsp(dist_m: np.ndarray) -> List: Cambio: Calcula todos los posibles circuitos y devuelve el circuito con la distancia más larga.
"""
def longest_exhaustive_tsp(dist_m: np.ndarray) -> List:
    """Calcula todos los posibles circuitos y devuelve el circuito con la distancia más larga"""
    n = len(dist_m)
    nodes = list(range(n))
    all_circuits = []

    def backtrack(circuit):
        if len(circuit) == n:
            all_circuits.append(circuit[:])
        else:
            for node in nodes:
                if node not in circuit:
                    circuit.append(node)
                    backtrack(circuit)
                    circuit.pop()

    backtrack([])
    longest_circuit = max(all_circuits, key=lambda c: len_circuit(c, dist_m))
    return longest_circuit

"""init_cd_with_weights(n: int, weights: List[int]) -> np.ndarray: Cambio: Inicializa el conjunto disjunto con pesos asignados a cada nodo.
"""
def init_cd_with_weights(n: int, weights: List[int]) -> np.ndarray:
    """Inicializa el conjunto disjunto con pesos asignados a cada nodo"""
    p_cd = np.arange(n)
    w_cd = np.array(weights)
    return np.stack((p_cd, w_cd), axis=1)



"""union_with_rank(rep_1: int, rep_2: int, p_cd: np.ndarray, ranks: np.ndarray) -> int: Cambio: Une dos nodos de un conjunto disjunto y mantiene un arreglo de rangos para optimizar la unión.
"""
def union_with_rank(rep_1: int, rep_2: int, p_cd: np.ndarray, ranks: np.ndarray) -> int:
    """Une dos nodos de un conjunto disjunto y mantiene un arreglo de rangos"""
    if ranks[rep_1] < ranks[rep_2]:
        p_cd[rep_1] = rep_2
    elif ranks[rep_1] > ranks[rep_2]:
        p_cd[rep_2] = rep_1
    else:
        p_cd[rep_2] = rep_1
        ranks[rep_1] += 1
    
    return p_cd[rep_1]
"""find_with_path_compression(ind: int, p_cd: np.ndarray) -> int: Cambio: Encuentra la raíz del nodo con compresión de ruta y devuelve el representante.
"""
def find_with_path_compression(ind: int, p_cd: np.ndarray) -> int:
    """Encuentra la raíz del nodo con compresión de ruta y devuelve el representante"""
    if p_cd[ind] != ind:
        p_cd[ind] = find_with_path_compression(p_cd[ind], p_cd)
    return p_cd[ind]


"""ccs_with_sizes(n: int, l: List) -> Dict: Cambio: Modifica la función ccs para devolver también el tamaño de cada componente conexa.
"""
def ccs_with_sizes(n: int, l: List) -> Dict:
    """Devuelve las componentes conexas de un grafo en forma de diccionario, incluyendo sus tamaños"""
    p_cd = init_cd(n)
    sizes = {i: 1 for i in range(n)}

    for edge in l:
        rep_1 = find(edge[0], p_cd)
        rep_2 = find(edge[1], p_cd)
        if rep_1 != rep_2:
            union(rep_1, rep_2, p_cd)
            sizes[rep_1] += sizes[rep_2]
            del sizes[rep_2]

    components = {}
    for i in range(n):
        rep = find(i, p_cd)
        if rep not in components:
            components[rep] = []
        components[rep].append(i)

    return components, sizes
"""greedy_tsp_with_start(dist_m: np.ndarray, start_node: int) -> List: Cambio: Modifica la función greedy_tsp para que acepte un nodo de inicio específico.
"""
def greedy_tsp_with_start(dist_m: np.ndarray, start_node: int) -> List:
    """Genera un circuito codicioso a partir de una matriz y un nodo de inicio específico"""
    n = len(dist_m)
    visited = [False] * n
    circuit = [start_node]
    visited[start_node] = True

    for _ in range(n - 1):
        current_node = circuit[-1]
        min_dist = float('inf')
        next_node = None

        for neighbor in range(n):
            if not visited[neighbor] and dist_m[current_node][neighbor] < min_dist:
                min_dist = dist_m[current_node][neighbor]
                next_node = neighbor

        circuit.append(next_node)
        visited[next_node] = True

    circuit.append(start_node)  # Vuelve al nodo de inicio para cerrar el circuito
    return circuit

"""find_with_path_compression(ind: int, p_cd: np.ndarray) -> int: Cambio: Modifica la función find para implementar la compresión de ruta.
"""
def find_with_path_compression(ind: int, p_cd: np.ndarray) -> int:
    """Encuentra la raíz del nodo ind y realiza compresión de ruta"""
    if p_cd[ind] != ind:
        p_cd[ind] = find_with_path_compression(p_cd[ind], p_cd)
    return p_cd[ind]
"""union_by_rank(rep_1: int, rep_2: int, p_cd: np.ndarray, rank: np.ndarray) -> int: Cambio: Modifica la función union para utilizar la unión por rango.
"""
def union_by_rank(rep_1: int, rep_2: int, p_cd: np.ndarray, rank: np.ndarray) -> int:
    """Realiza la unión de dos nodos utilizando la unión por rango"""
    if rank[rep_1] < rank[rep_2]:
        p_cd[rep_1] = rep_2
        return rep_2
    elif rank[rep_1] > rank[rep_2]:
        p_cd[rep_2] = rep_1
        return rep_1
    else:
        p_cd[rep_2] = rep_1
        rank[rep_1] += 1
        return rep_1
"""dist_matrix_symmetric(n_nodes: int, w_max=10) -> np.ndarray: Cambio: Modifica la función dist_matrix para generar una matriz de distancia simétrica.
"""
def dist_matrix_symmetric(n_nodes: int, w_max=10) -> np.ndarray:
    """Crea una matriz de distancia aleatoria simétrica"""
    dist_m = np.zeros((n_nodes, n_nodes))

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            weight = np.random.randint(1, w_max + 1)
            dist_m[i][j] = weight
            dist_m[j][i] = weight

    return dist_m

"""cd_2_dict_with_size(p_cd: np.ndarray, size: np.ndarray) -> Dict: Cambio: Modifica la función cd_2_dict para incluir el tamaño de cada conjunto en el diccionario.
"""
def cd_2_dict_with_size(p_cd: np.ndarray, size: np.ndarray) -> Dict:
    """Crea un diccionario a partir de un conjunto disjunto, incluyendo el tamaño de cada conjunto"""
    dict_cd = {}
    for i, rep in enumerate(p_cd):
        root = find(i, p_cd)
        if root in dict_cd:
            dict_cd[root].append(i)
        else:
            dict_cd[root] = [i]
    for rep, nodes in dict_cd.items():
        dict_cd[rep] = {"nodes": nodes, "size": size[rep]}
    return dict_cd
"""greedy_tsp_random_start(dist_m: np.ndarray) -> List: Cambio: Modifica la función greedy_tsp para seleccionar aleatoriamente el nodo de inicio.
"""
def greedy_tsp_random_start(dist_m: np.ndarray) -> List:
    """Genera un circuito codicioso a partir de una matriz y un nodo de inicio seleccionado aleatoriamente"""
    n_nodes = dist_m.shape[0]
    start_node = np.random.randint(n_nodes)
    return greedy_tsp(dist_m, start_node)
"""repeated_greedy_tsp_random_start(dist_m: np.ndarray) -> List: Cambio: Modifica la función repeated_greedy_tsp para utilizar greedy_tsp_random_start en lugar de greedy_tsp.
"""
def repeated_greedy_tsp_random_start(dist_m: np.ndarray) -> List:
    """Calcula el mejor camino encontrado por greedy_tsp repetido por el número de nodos de la matriz, con inicio aleatorio"""
    n_nodes = dist_m.shape[0]
    best_circuit = None
    best_length = float('inf')
    for _ in range(n_nodes):
        circuit = greedy_tsp_random_start(dist_m)
        length = len_circuit(circuit, dist_m)
        if length < best_length:
            best_circuit = circuit
            best_length = length
    return best_circuit


"""weighted_union(rep_1: int, rep_2: int, p_cd: np.ndarray, size: np.ndarray, weights: np.ndarray) -> int: Cambio: Modifica la función union para realizar una unión ponderada de los conjuntos, teniendo en cuenta los pesos de los elementos.
"""
def weighted_union(rep_1: int, rep_2: int, p_cd: np.ndarray, size: np.ndarray, weights: np.ndarray) -> int:
    """
    Esta función une dos nodos de un conjunto disjunto considerando los pesos de los elementos.
    Devuelve el representante del conjunto obtenido.
    """
    root_1 = find(rep_1, p_cd)
    root_2 = find(rep_2, p_cd)

    if root_1 != root_2:
        if size[root_1] > size[root_2]:
            p_cd[root_2] = root_1
            size[root_1] += size[root_2]
            weights[root_1] += weights[root_2]
        else:
            p_cd[root_1] = root_2
            size[root_2] += size[root_1]
            weights[root_2] += weights[root_1]

    return find(rep_1, p_cd)

"""find_with_path_compression(ind: int, p_cd: np.ndarray) -> int: Cambio: Modifica la función find para realizar compresión de ruta en la búsqueda del representante.
"""
def find_with_path_compression(ind: int, p_cd: np.ndarray) -> int:
    """
    Encuentra la raíz del nodo ind con compresión de ruta.
    Devuelve el representante.
    """
    if p_cd[ind] != ind:
        p_cd[ind] = find_with_path_compression(p_cd[ind], p_cd)
    return p_cd[ind]

"""kruskal_mst(edges: List[Tuple[int, int, int]], n_nodes: int) -> List[Tuple[int, int, int]]: Cambio: Implementa el algoritmo de Kruskal para encontrar el árbol de expansión mínima (MST) en un grafo dado.
"""
def kruskal_mst(edges: List[Tuple[int, int, int]], n_nodes: int) -> List[Tuple[int, int, int]]:
    """
    Implementación del algoritmo de Kruskal para encontrar el árbol de expansión mínima (MST).
    Devuelve las aristas del MST.
    """
    edges.sort(key=lambda x: x[2])  # Ordenar aristas por peso
    p_cd = np.arange(n_nodes)
    mst = []
    for edge in edges:
        u, v, weight = edge
        if find(u, p_cd) != find(v, p_cd):
            union(u, v, p_cd)
            mst.append(edge)
    return mst

"""##############################################################################################################################################################"""
"""##############################################################################################################################################################"""
"""##############################################################################################################################################################"""
"""##############################################################################################################################################################"""

"""Practica 3
    """

"""pivot5(t: List, f: int, l: int) - Cambio: Encontrar el pivote como el elemento máximo en lugar del elemento medio.
"""
from typing import List

def pivot_max(t: List, f: int, l: int):
    """Encuentra el pivote como el elemento máximo en una lista"""
    return l


"""QSelect(t: List, f: int, l: int, k: int) - Cambio: Utilizar el pivote máximo en lugar del pivote medio.
"""
from typing import List

def QSelect_max(t: List, f: int, l: int, k: int):
    """Selecciona el k-ésimo elemento en una lista utilizando el pivote máximo"""
    if f == l:
        return t[f]

    pivot_index = pivot_max(t, f, l)
    pivot = t[pivot_index]
    t[pivot_index], t[l] = t[l], t[pivot_index]

    i = f
    for j in range(f, l):
        if t[j] > pivot:
            t[i], t[j] = t[j], t[i]
            i += 1

    t[i], t[l] = t[l], t[i]

    if k == i:
        return t[i]
    elif k < i:
        return QSelect_max(t, f, i - 1, k)
    else:
        return QSelect_max(t, i + 1, l, k)
    

"""split_random_pivot(t: np.ndarray): Cambio: Modifica la función split para utilizar un pivote seleccionado aleatoriamente en lugar del primer elemento.
"""
def split_random_pivot(t: np.ndarray):
    """
    Divide una lista en dos partes: elementos menores y mayores que el pivote (seleccionado aleatoriamente).
    Devuelve una tupla con los elementos menores, el pivote y los elementos mayores.
    """
    piv = np.random.choice(t)
    menores = []
    mayores = []
    for elem in t:
        if elem < piv:
            menores.append(elem)
        elif elem > piv:
            mayores.append(elem)
    return (menores, piv, mayores)
"""qsel_random_pivot(t: np.ndarray, k: int): Cambio: Modifica la función qsel para utilizar el pivote seleccionado aleatoriamente en la función split.
"""
def qsel_random_pivot(t: np.ndarray, k: int):
    """
    Implementación de quick select utilizando un pivote seleccionado aleatoriamente en la función split.
    Devuelve el k-ésimo elemento de la lista ordenada.
    """
    if len(t) == 0:
        return None
    menores, piv, mayores = split_random_pivot(t)
    if k < len(menores):
        return qsel_random_pivot(menores, k)
    elif k > len(menores):
        return qsel_random_pivot(mayores, k - len(menores) - 1)
    else:
        return piv
"""qsel_nr_random_pivot(t: np.ndarray, k: int): Cambio: Modifica la función qsel_nr para utilizar el pivote seleccionado aleatoriamente en la función split.
"""
def qsel_nr_random_pivot(t: np.ndarray, k: int):
    """
    Implementación de quick select no recursiva utilizando un pivote seleccionado aleatoriamente en la función split.
    Devuelve el k-ésimo elemento de la lista ordenada.
    """
    if len(t) == 0:
        return None
    while True:
        menores, piv, mayores = split_random_pivot(t)
        if k < len(menores):
            t = menores
        elif k > len(menores):
            t = mayores
            k = k - len(menores) - 1
        else:
            return piv

"""qsel_nr_median_pivot(t: np.ndarray, k: int): Cambio: Modifica la función qsel_nr para utilizar la mediana de la lista como pivote en la función split.
"""
def qsel_nr_median_pivot(t: np.ndarray, k: int):
    """
    Implementación de quick select no recursiva utilizando la mediana como pivote en la función split.
    Devuelve el k-ésimo elemento de la lista ordenada.
    """
    if len(t) == 0:
        return None
    while True:
        piv = np.median(t)
        menores, _, mayores = split_pivot(t, piv)
        if k < len(menores):
            t = menores
        elif k > len(menores):
            t = mayores
            k = k - len(menores) - 1
        else:
            return piv
"""edit_distance_recursive(str_1: str, str_2: str): Cambio: Modifica la función edit_distance para implementar un enfoque recursivo en lugar de la programación dinámica.
"""
def edit_distance_recursive(str_1: str, str_2: str):
    """
    Calcula la distancia de edición entre dos cadenas utilizando un enfoque recursivo.
    Devuelve la distancia de edición.
    """
    if len(str_1) == 0:
        return len(str_2)
    if len(str_2) == 0:
        return len(str_1)
    if str_1[0] == str_2[0]:
        return edit_distance_recursive(str_1[1:], str_2[1:])
    else:
        insert = 1 + edit_distance_recursive(str_1, str_2[1:])
        delete = 1 + edit_distance_recursive(str_1[1:], str_2)
        replace = 1 + edit_distance_recursive(str_1[1:], str_2[1:])
        return min(insert, delete, replace)

"""qsel_nr_random_pivot(t: np.ndarray, k: int): Cambio: Modifica la función qsel_nr para seleccionar un pivote aleatorio en la función split.
"""

import random

def qsel_nr_random_pivot(t: np.ndarray, k: int):
    """
    Implementación de quick select no recursiva utilizando un pivote aleatorio en la función split.
    Devuelve el k-ésimo elemento de la lista ordenada.
    """
    if len(t) == 0:
        return None
    while True:
        piv = random.choice(t)
        menores, _, mayores = split_pivot(t, piv)
        if k < len(menores):
            t = menores
        elif k > len(menores):
            t = mayores
            k = k - len(menores) - 1
        else:
            return piv
"""max_common_subsequence_recursive(str_1: str, str_2: str): Cambio: Modifica la función max_common_subsequence para implementar un enfoque recursivo en lugar de la programación dinámica.
"""
def max_common_subsequence_recursive(str_1: str, str_2: str):
    """
    Encuentra la subsecuencia común más larga entre dos cadenas utilizando un enfoque recursivo.
    Devuelve la subsecuencia común más larga.
    """
    if len(str_1) == 0 or len(str_2) == 0:
        return ""
    if str_1[-1] == str_2[-1]:
        return max_common_subsequence_recursive(str_1[:-1], str_2[:-1]) + str_1[-1]
    else:
        subseq1 = max_common_subsequence_recursive(str_1[:-1], str_2)
        subseq2 = max_common_subsequence_recursive(str_1, str_2[:-1])
        if len(subseq1) > len(subseq2):
            return subseq1
        else:
            return subseq2


"""qsel_recursive(t: np.ndarray, k: int): Cambio: Modifica la función qsel para implementar un enfoque recursivo en lugar de la iteración.
"""
def qsel_recursive(t: np.ndarray, k: int):
    """
    Implementación de quick select utilizando un enfoque recursivo en lugar de la iteración.
    Devuelve el k-ésimo elemento de la lista ordenada.
    """
    if len(t) == 0:
        return None
    pivot = random.choice(t)
    smaller = [x for x in t if x < pivot]
    equal = [x for x in t if x == pivot]
    larger = [x for x in t if x > pivot]
    if k < len(smaller):
        return qsel_recursive(smaller, k)
    elif k < len(smaller) + len(equal):
        return pivot
    else:
        return qsel_recursive(larger, k - len(smaller) - len(equal))
"""edit_distance_recursive(str_1: str, str_2: str): Cambio: Modifica la función edit_distance para implementar un enfoque recursivo en lugar de la programación dinámica.
"""
def edit_distance_recursive(str_1: str, str_2: str):
    """
    Calcula la distancia de edición entre dos cadenas utilizando un enfoque recursivo.
    Devuelve la distancia de edición.
    """
    if len(str_1) == 0:
        return len(str_2)
    if len(str_2) == 0:
        return len(str_1)
    if str_1[-1] == str_2[-1]:
        return edit_distance_recursive(str_1[:-1], str_2[:-1])
    else:
        substitution = edit_distance_recursive(str_1[:-1], str_2[:-1]) + 1
        insertion = edit_distance_recursive(str_1, str_2[:-1]) + 1
        deletion = edit_distance_recursive(str_1[:-1], str_2) + 1
        return min(substitution, insertion, deletion)
"""
split_random_pivot(t: np.ndarray): Cambio: Modifica la función split para seleccionar el pivote de manera aleatoria en lugar de usar el primer elemento.
"""
def split_random_pivot(t: np.ndarray):
    """
    Divide una lista en tres partes: elementos menores que el pivote, el pivote y elementos mayores que el pivote.
    El pivote se selecciona de manera aleatoria.
    Devuelve una tupla con las listas de elementos menores, el pivote y elementos mayores.
    """
    pivot = random.choice(t)
    less = [x for x in t if x < pivot]
    equal = [x for x in t if x == pivot]
    greater = [x for x in t if x > pivot]
    return (less, equal, greater)
"""qsel_nr_random_pivot(t: np.ndarray, k: int): Cambio: Modifica la función qsel_nr para utilizar la función split_random_pivot en lugar de la función split.
"""
def qsel_nr_random_pivot(t: np.ndarray, k: int):
    """
    Implementación de quick select no recursiva utilizando un pivote seleccionado de manera aleatoria.
    Devuelve el k-ésimo elemento de la lista ordenada.
    """
    stack = [(0, len(t)-1)]
    while stack:
        left, right = stack.pop()
        if left >= right:
            return t[left]
        pivot = random.choice(t[left:right+1])
        i, j = left, right
        while i <= j:
            while t[i] < pivot:
                i += 1
            while t[j] > pivot:
                j -= 1
            if i <= j:
                t[i], t[j] = t[j], t[i]
                i += 1
                j -= 1
        if k <= j:
            stack.append((left, j))
        elif k >= i:
            stack.append((i, right))
            k -= i
        else:
            return t[k]

"""qsel_pivot_median(t: np.ndarray, k: int): Cambio: Modifica la función qsel para seleccionar el pivote como la mediana de los elementos de la lista.
"""
def qsel_pivot_median(t: np.ndarray, k: int):
    """
    Implementación de quick select utilizando la mediana como pivote.
    Devuelve el k-ésimo elemento de la lista ordenada.
    """
    pivot = np.median(t)
    less = [x for x in t if x < pivot]
    equal = [x for x in t if x == pivot]
    greater = [x for x in t if x > pivot]
    if k <= len(less):
        return qsel_pivot_median(less, k)
    elif k <= len(less) + len(equal):
        return pivot
    else:
        return qsel_pivot_median(greater, k - len(less) - len(equal))
"""qsel_nr_random_partition(t: np.ndarray, k: int): Cambio: Modifica la función qsel_nr para utilizar una partición aleatoria en lugar de la partición basada en el primer elemento.
"""
def qsel_nr_random_partition(t: np.ndarray, k: int):
    """
    Implementación de quick select no recursiva utilizando una partición aleatoria.
    Devuelve el k-ésimo elemento de la lista ordenada.
    """
    stack = [(0, len(t)-1)]
    while stack:
        left, right = stack.pop()
        if left >= right:
            return t[left]
        pivot_index = random.randint(left, right)
        pivot = t[pivot_index]
        t[pivot_index], t[right] = t[right], t[pivot_index]
        i = left
        for j in range(left, right):
            if t[j] < pivot:
                t[i], t[j] = t[j], t[i]
                i += 1
        t[i], t[right] = t[right], t[i]
        if k <= i:
            stack.append((left, i-1))
        elif k >= i+2:
            stack.append((i+1, right))
            k -= i+1
        else:
            return t[i]

"""qsel_nr_descending(t: np.ndarray, k: int): Cambio: Modifica la función qsel_nr para seleccionar los elementos en orden descendente en lugar de ascendente.
"""
def qsel_nr_descending(t: np.ndarray, k: int):
    """
    Implementación de quick select no recursiva que devuelve el k-ésimo elemento más grande en lugar del más pequeño.
    """
    stack = [(0, len(t)-1)]
    while stack:
        left, right = stack.pop()
        if left >= right:
            return t[left]
        pivot = t[right]
        i = left
        for j in range(left, right):
            if t[j] >= pivot:
                t[i], t[j] = t[j], t[i]
                i += 1
        t[i], t[right] = t[right], t[i]
        if k <= i:
            stack.append((left, i-1))
        elif k >= i+2:
            stack.append((i+1, right))
            k -= i+1
        else:
            return t[i]
"""split_random(t: np.ndarray): Cambio: Modifica la función split para realizar la partición utilizando un índice aleatorio en lugar del primer elemento.
python
Copy code"""
def split_random(t: np.ndarray):
    """
    Particiona la lista en dos subconjuntos según un índice aleatorio.
    Devuelve una tupla con los elementos menores, el pivote y los elementos mayores.
    """
    pivot_index = random.randint(0, len(t)-1)
    pivot = t[pivot_index]
    less = [x for i, x in enumerate(t) if x < pivot or (x == pivot and i < pivot_index)]
    greater = [x for i, x in enumerate(t) if x > pivot or (x == pivot and i > pivot_index)]
    return (less, pivot, greater)
"""
qsel_random(t: np.ndarray, k: int): Cambio: Modifica la función qsel para seleccionar un elemento aleatorio en lugar de utilizar el primer elemento como pivote.
python
Copy code"""
def qsel_random(t: np.ndarray, k: int):
    """
    Implementación de quick select que utiliza un pivote aleatorio en lugar del primer elemento.
    Devuelve el k-ésimo elemento más pequeño.
    """
    if len(t) == 1:
        return t[0]
    pivot_index = random.randint(0, len(t)-1)
    pivot = t[pivot_index]
    less = [x for x in t if x < pivot]
    greater = [x for x in t if x > pivot]
    equal = [x for x in t if x == pivot]
    if k < len(less):
        return qsel_random(less, k)
    elif k < len(less) + len(equal):
        return equal[0]
    else:
        return qsel_random(greater, k - len(less) - len(equal))
"""split_recursive(t: np.ndarray): Cambio: Modifica la función split para utilizar una implementación recursiva en lugar de un bucle.
python
Copy code"""
def split_recursive(t: np.ndarray):
    """
    Particiona la lista en dos subconjuntos utilizando una implementación recursiva.
    Devuelve una tupla con los elementos menores, el pivote y los elementos mayores.
    """
    if len(t) <= 1:
        return ([], t[0], [])
    pivot = t[0]
    less = [x for x in t[1:] if x < pivot]
    greater = [x for x in t[1:] if x >= pivot]
    return (less, pivot, greater)

"""qsel_reverse(t: np.ndarray, k: int): Cambio: Modifica la función qsel para seleccionar el k-ésimo elemento más grande en lugar del k-ésimo elemento más pequeño.
python
Copy code"""
def qsel_reverse(t: np.ndarray, k: int):
    """
    Implementación de quick select que devuelve el k-ésimo elemento más grande en lugar del más pequeño.
    """
    if len(t) == 1:
        return t[0]
    pivot_index = random.randint(0, len(t)-1)
    pivot = t[pivot_index]
    less = [x for x in t if x <= pivot]
    greater = [x for x in t if x > pivot]
    if k <= len(greater):
        return qsel_reverse(greater, k)
    else:
        return qsel_reverse(less, k - len(greater))
"""edit_distance_recursive(str_1: str, str_2: str): Cambio: Modifica la función edit_distance para utilizar una implementación recursiva en lugar de la programación dinámica.
python
Copy code"""
def edit_distance_recursive(str_1: str, str_2: str):
    """
    Calcula la distancia de edición entre dos cadenas utilizando una implementación recursiva.
    """
    if len(str_1) == 0:
        return len(str_2)
    if len(str_2) == 0:
        return len(str_1)
    if str_1[0] == str_2[0]:
        return edit_distance_recursive(str_1[1:], str_2[1:])
    else:
        return 1 + min(
            edit_distance_recursive(str_1[1:], str_2),
            edit_distance_recursive(str_1, str_2[1:]),
            edit_distance_recursive(str_1[1:], str_2[1:])
        )

"""qsel_random_pivot(t: np.ndarray, k: int): Cambio: Modifica la función qsel para seleccionar un pivote aleatorio en lugar de siempre tomar el primer elemento como pivote.
python
Copy code"""
def qsel_random_pivot(t: np.ndarray, k: int):
    """
    Implementación de quick select que utiliza un pivote aleatorio en lugar del primer elemento.
    """
    if len(t) == 1:
        return t[0]
    pivot_index = random.randint(0, len(t)-1)
    pivot = t[pivot_index]
    less = [x for x in t if x < pivot]
    equal = [x for x in t if x == pivot]
    greater = [x for x in t if x > pivot]
    if k <= len(less):
        return qsel_random_pivot(less, k)
    elif k <= len(less) + len(equal):
        return equal[0]
    else:
        return qsel_random_pivot(greater, k - len(less) - len(equal))
"""split_pivot_random(t: np.ndarray): Cambio: Modifica la función split_pivot para seleccionar un pivote aleatorio en lugar de siempre tomar el elemento del medio como pivote.
python
Copy code"""
def split_pivot_random(t: np.ndarray):
    """
    Divide el arreglo en menores, pivote y mayores utilizando un pivote aleatorio.
    """
    piv = random.choice(t)
    less = [x for x in t if x < piv]
    equal = [x for x in t if x == piv]
    greater = [x for x in t if x > piv]
    return (less, equal, greater)
 
"""qsel_random_partition(t: np.ndarray, k: int): Cambio: Modifica la función qsel para realizar particiones aleatorias en lugar de particiones determinísticas.
python
Copy code"""
def qsel_random_partition(t: np.ndarray, k: int):
    """
    Implementación de quick select que utiliza particiones aleatorias en lugar de particiones determinísticas.
    """
    if len(t) == 1:
        return t[0]
    pivot_index = random.randint(0, len(t)-1)
    pivot = t[pivot_index]
    t_less = []
    t_equal = []
    t_greater = []
    for x in t:
        if x < pivot:
            t_less.append(x)
        elif x == pivot:
            t_equal.append(x)
        else:
            t_greater.append(x)
    if k <= len(t_less):
        return qsel_random_partition(t_less, k)
    elif k <= len(t_less) + len(t_equal):
        return t_equal[0]
    else:
        return qsel_random_partition(t_greater, k - len(t_less) - len(t_equal))
"""split_pivot_median_of_three(t: np.ndarray): Cambio: Modifica la función split_pivot para seleccionar el pivote como la mediana de tres elementos aleatorios en lugar del elemento del medio.
python
Copy code"""
def split_pivot_median_of_three(t: np.ndarray):
    """
    Divide el arreglo en menores, pivote y mayores utilizando el pivote como la mediana de tres elementos aleatorios.
    """
    random_indices = random.sample(range(len(t)), 3)
    random_values = [t[i] for i in random_indices]
    piv = statistics.median(random_values)
    less = [x for x in t if x < piv]
    equal = [x for x in t if x == piv]
    greater = [x for x in t if x > piv]
    return (less, equal, greater)

"""qsel_random_partition_recursive(t: np.ndarray, k: int): Cambio: Modifica la función qsel para realizar particiones aleatorias de forma recursiva.
python
Copy code
"""
def qsel_random_partition_recursive(t: np.ndarray, k: int):
    """
    Implementación de quick select que utiliza particiones aleatorias de forma recursiva.
    """
    if len(t) == 1:
        return t[0]
    pivot_index = random.randint(0, len(t)-1)
    pivot = t[pivot_index]
    t_less = [x for x in t if x < pivot]
    t_equal = [x for x in t if x == pivot]
    t_greater = [x for x in t if x > pivot]
    if k <= len(t_less):
        return qsel_random_partition_recursive(t_less, k)
    elif k <= len(t_less) + len(t_equal):
        return t_equal[0]
    else:
        return qsel_random_partition_recursive(t_greater, k - len(t_less) - len(t_equal))
"""split_pivot_random(t: np.ndarray): Cambio: Modifica la función split_pivot para seleccionar el pivote aleatoriamente en lugar de utilizar un elemento fijo.
python
Copy code"""
def split_pivot_random(t: np.ndarray):
    """
    Divide el arreglo en menores, pivote y mayores utilizando un pivote seleccionado aleatoriamente.
    """
    pivot_index = random.randint(0, len(t)-1)
    pivot = t[pivot_index]
    less = [x for x in t if x < pivot]
    equal = [x for x in t if x == pivot]
    greater = [x for x in t if x > pivot]
    return (less, equal, greater)
"""
qsel_iterative(t: np.ndarray, k: int): Cambio: Modifica la función qsel para implementar una versión iterativa en lugar de una versión recursiva.
python
Copy code
"""
def qsel_iterative(t: np.ndarray, k: int):
    """
    Implementación de quick select de forma iterativa.
    """
    left = 0
    right = len(t) - 1

    while True:
        pivot_index = random.randint(left, right)
        pivot = t[pivot_index]
        t[left], t[pivot_index] = t[pivot_index], t[left]
        i = left + 1
        j = right

        while i <= j:
            if t[i] > pivot and t[j] < pivot:
                t[i], t[j] = t[j], t[i]
            if t[i] <= pivot:
                i += 1
            if t[j] >= pivot:
                j -= 1

        t[left], t[j] = t[j], t[left]

        if k == j:
            return t[j]
        elif k < j:
            right = j - 1
        else:
            left = j + 1
"""split_pivot_randomized(t: np.ndarray): Cambio: Modifica la función split_pivot para seleccionar el pivote aleatoriamente utilizando una técnica de partición aleatoria.
"""
def split_pivot_randomized(t: np.ndarray):
    """
    Divide el arreglo en menores, pivote y mayores utilizando la técnica de partición aleatoria.
    """
    pivot_index = random.randint(0, len(t) - 1)
    pivot = t[pivot_index]
    t_less = []
    t_equal = []
    t_greater = []

    for x in t:
        if x < pivot:
            t_less.append(x)
        elif x == pivot:
            t_equal.append(x)
        else:
            t_greater.append(x)

    return (t_less, t_equal, t_greater)