import sys
import math
import numpy as np
import itertools

from typing import List, Dict, Callable, Iterable, Tuple
#from p214 import *

# Exercise 1A


def init_cd(n: int) -> np.ndarray:
    """  Inicializa un conjunto disjunto

        Argumentos: 

            n(int): numero de posiciones del CD

        Devuelve:

            np.full(n, -1)(np.ndarray): un array de n posiciones lleno de -1

        Autor:

            Eduardo

    """
    return np.full(n, -1)


def union(rep_1: int, rep_2: int, p_cd: np.ndarray) -> int:
    """  Une dos conjuntos disjuntos

        Argumentos:

            rep_1(int): representante del primer subconjunto
            rep_2(int): representante del segundo subconjunto
            p_cd(np.ndarray): array del CD

        Devuelve:

            p_cd(np.ndarray): el array del CD con la unión de los subconjuntos

        Autor:

            Eduardo y Dimitris

    """
    p_cd[rep_2-1] = rep_1  # join second tree to first
    return p_cd


def find(ind: int, p_cd: np.ndarray) -> int:
    """  Busca el representante de un CD

        Argumentos:

            ind(int): índice del cual se busca su representante
            p_cd(np.ndarray): array del CD

        Devuelve:

            p_c[ind-n+1](int): el elemento el cual representa al índice dado

        Autor:

            Eduardo

    """
    n = 1
    while p_cd[ind-n] > 0:
        n = n+1
    return p_cd[ind-n+1]


def cd_2_dict(p_cd: np.ndarray) -> dict:
    """  Imprime un diccionario del CD

        Argumentos:

            p_cd(np.ndarray): array del CD

        Devuelve:

            p_cd(np.ndarray): el array del CD con la unión de los subconjuntos

        Autor:

            Dimitris

    """
    d = {}
    i = 0
    u = find(i, p_cd)
    j = 0
    k = 0

    for i in range(len(p_cd)):
        if p_cd[i] > 0:
            u = find(i, p_cd)
            d[u] = p_cd[i]
            k = p_cd[i]
            print(d)
            j = j-1
        elif p_cd[i] < 0:
            if j > 0:
                k = k+1
                d[u] = k
                print(d)

            if p_cd[i] == -1:
                k = k+1
                d[k] = k
                j = 0
                print(d)
                continue

            j = p_cd[i]*(-1)
    if j > 0:
        k = k+1
        d[u] = k
    return d

# Exercise 1B


def ccs(n: int, l: List) -> dict:
    """  Devuelve las componentes conexas de un tal grafo.

        Argumentos:

            n(int): número de vértices del grafo
            l(list): la lista que describe las aristas del grafo

        Devuelve:

            graph(dict): las componentes conexas del grafo

        Autor:

            Eduardo

    """
    a = init_cd(n)
    for u, v, in l:
        r_u = find(u, a)
        r_v = find(v, a)
        if r_u != r_v:
            union(r_u, r_v, a)
    d_cc = cd_2_dict(a)

    return d_cc

# Ecercise 2A


def dist_matrix(n_nodes: int, w_max=10) -> np.ndarray:
    """  Genera la matriz distrancia de un grafo

        Argumentos:

            n_nodes(int): número de nodos del grafo
            w_max(10): máximo valor del los nodos del grafo

        Devuelve:

            dist_m(np.ndarray): la matriz distancia del grafo dado

        Autor:

            Eduardo

    """

    dist_m = np.random.randint(0, w_max, (n_nodes, n_nodes))
    dist_m = (dist_m + dist_m.T) // 2
    dist_m = dist_m - np.diag(np.diag(dist_m))

    return dist_m


def greedy_tsp(dist_m: np.ndarray, node_ini=0) -> List:
    """  Algoritmo codicioso que encuentra el circuito óptimo de un grafo

        Argumentos:

            dist_m(np.ndarray): la matriz distancia del grafo dado
            node_ini(0): nodo inicial del que parte a realizar la búsqueda

        Devuelve:

            circuit + [node_ini]: el circuito más corto desde el nodo inicial

        Autor:

            Eduardo

    """

    num_cities = dist_m.shape[0]
    circuit = [node_ini]

    while len(circuit) < num_cities:
        current_city = circuit[-1]
        options = list(np.argsort(dist_m[current_city]))
        for city in options:
            if city not in circuit:
                circuit.append(city)
                break

    return circuit + [node_ini]


def len_circuit(circuit: List, dist_m: np.ndarray) -> int:
    """  Determina la longitud de un circuito a partir de una matriz distancia

        Argumentos:

            circuit(list): el circuto del cual se quiere saber la distancia
            dist_m(np.ndarray): la matriz distancia del grafo dado

        Devuelve:

            dist(int): la distancia del circuito 

        Autor:

            Eduardo

    """

    dist = 0
    i = 0

    for i in range(len(dist_m)):
        dist += int(dist_m[circuit[i], circuit[i+1]])

    return dist


def repeated_greedy_tsp(dist_m: np.ndarray) -> List:
    """  Algoritmo codicioso que encuentra el circuito óptimo de un grafo 
        aplicando la función greedy_tsp a cada nodo del grafo

        Argumentos:

            dist_m(np.ndarray): la matriz distancia del grafo dado

        Devuelve:

            minim(list): el circuito más corto encontrado

        Autor:

            Eduardo

    """
    i = 0
    minim = greedy_tsp(dist_m, i)

    for i in range(len(dist_m)):
        if (minim < greedy_tsp(dist_m, i-1)):
            minim = greedy_tsp(dist_m, i)

    return minim


def exhaustive_tsp(dist_m: np.ndarray) -> List:
    """  Algoritmo exhaustivo que que examina todos los posibles circuitos 
        y encuentra aquel de distancia más corta

        Argumentos:

            dist_m(np.ndarray): la matriz distancia del grafo dado

        Devuelve:

            minim(list): el circuito más corto encontrado

        Autor:

            Eduardo

    """

    int_max = 2**128
    len_min = int_max
    for perm in itertools.permutations(range(len(dist_m))):
        new_len = len_circuit(list(perm) + [list(perm)[0]], dist_m)
        if new_len < len_min:
            len_min = new_len

    return list(perm) + [list(perm)[0]]
