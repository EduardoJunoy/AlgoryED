#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import argparse
import textwrap

import numpy as np

from typing import Tuple

#import p100 as p1
  
####################################### main
def max_heapify(h: np.ndarray, i: int):
    """
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
    """
    """
    for i in range(len(h) // 2 - 1, -1, -1):
        max_heapify(h, i)

def extract_max_heap(h: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    """
    

    pass
    

def sort_max_heap(t: np.ndarray) -> np.ndarray:
    """
    """
    pass
    

def main(t_size: int):
    """
    """
    for _ in range(5):
        t = np.random.permutation(t_size)
        
        tt = t.copy()
        print('t        ', tt)
        
        #chequear create min heaps
        create_max_heap(tt)
        print('max heap?', tt)
        
        #chequear ordenacion
        tt = sort_max_heap(tt)
        print('sorted   ', tt, '\n')
    
      
###############################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
        """
        Examen de prácticas extraordinario 2023.
        """))
    
    parser.add_argument("-s", "--size", help="size of arrays", type=int, default=5)
    
    args = parser.parse_args()
    
    main(args.size)