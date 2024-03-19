    
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


def mochila(n,c):
    if n == 0 or c == 0:
        # solucion optima para cuando no quedan elementos o la capacidad disponible es 0
        return 0
    elif datos[n].peso > c:
        # no metemos el elemento
        return mochila(n-1,c)
    else:
        #sin meter el elemento
        a = mochila(n-1,c)
        # metiendo el elemento
        b = datos[n].valor + mochila(n-1,c-datos[n].peso)
        return max(a,b)


#V = list()
def mochila_rec(n,c):
    for x in range(n):
        V.append([])
        for y in range(c+1):
            V[x].append(0)
    for i in range(n):
        for w in range(c+1):
            if datos[i].peso <= w:
                V[i][w] = max(V[i-1][w],datos[i].valor+V[i-1][w-datos[i].peso])
            else:
                V[i][w] = V[i-1][w]
    
    return V[n-1][c]

#HUFFMAN

def get_probabilities(content):
    total = len(content) + 1 # Agregamos uno por el caracter FINAL
    c = Counter(content)
    res = {}
    for char,count in c.items():
        res[char] = float(count)/total
    res['end'] = 1.0/total
    return res

def make_tree(probs):
    q = []
    # Agregamos todos los símbolos a la pila
    for ch,pr in probs.items():
        # La fila de prioridad está ordenada por
        # prioridad y profundidad
        heapq.heappush(q,(pr,0,ch))

    # Empezamos a mezclar símbolos juntos
    # hasta que la fila tenga un elemento
    while len(q) > 1:
        e1 = heapq.heappop(q) # El símbolo menos probable
        e2 = heapq.heappop(q) # El segundo menos probable
        
        # Este nuevo nodo tiene probabilidad e1[0]+e2[0]
        # y profundidad mayor al nuevo nodo
        nw_e = (e1[0]+e2[0],max(e1[1],e2[1])+1,[e1,e2])
        heapq.heappush(q,nw_e)
    return q[0] # Devolvemos el arbol sin la fila

def make_dictionary(tree):
    res = {} # La estructura que vamos a devolver
    search_stack = [] # Pila para DFS
    # El último elemento de la lista es el prefijo!
    search_stack.append(tree+("",)) 
    while len(search_stack) > 0:
        elm = search_stack.pop()
        if type(elm[2]) == list:
            # En este caso, el nodo NO es una hoja del árbol,
            # es decir que tiene nodos hijos
            
            # El hijo izquierdo tiene "0" en el prefijo
            search_stack.append(elm[2][1]+(prefix+"0",))
            # El hijo derecho tiene "1" en el prefijo
            search_stack.append(elm[2][0]+(prefix+"1",))
            continue
        else:
            # El nodo es una hoja del árbol, así que
            # obtenemos el código completo y lo agregamos
            code = elm[-1]
            res[elm[2]] = code
        pass
    return res

def compress(dic,content):
    res = ""
    # Iteramos sobre cada elemento del archivo de entrada
    for ch in content:
        code = dic[ch]
        res = res + code
    # Agregamos el 1 a la izquierda, y el marcador de final
    # a la derecha
    res = '1' + res + dic['end']
    # Agregamos ceros para que la longitud del resultado
    # sea un múltiplo de 8
    res = res + (len(res) % 8 * "0")
    return int(res,2) # Convertimos a entero! (2 porque es base 2)

def store(data,dic,outfile):
    # Guardamos la cadena de bits en un archivo, que abrimos
    # en modo binario (por eso 'wb')
    outf = open(outfile,'wb')
    pickle.dump(compressed,outf)
    outf.close()

    # Guardamos el diccionario en otro archivo en formato JSON
    outf = open(outfile+".dic",'w')
    json.dump(dic,outf)
    outf.close()
    pass
    # Leemos el archivo de entrada completo a cont
    inf = open(sys.argv[1])
    cont = inf.read()
    inf.close()
    # Calculamos la distribución de probabilidad para cada símbolo
    probs = get_probabilities(cont)
    # Construimos el árbol de parseo! : )
    tree = make_tree(probs)
    # Construimos el diccionario para codificar
    dic = make_dictionary(tree)
    # Codificamos el contenido del archivo
    compressed = compress(dic,cont)
    # Guardamos todo en disco!
    store(compressed,dic,sys.argv[2])

    print("Archivo comprimido!")


