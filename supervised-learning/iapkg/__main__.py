#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 07:54:07 2021

@author: Jeremy Levens
"""

import time
from timeit import default_timer as timer
from multiprocessing import Pool, cpu_count


def power(x, n):

    time.sleep(1)

    return x ** n


def main():

    start = timer()

    print(f'starting computations on {cpu_count()} cores')

    values = ((2, 2), (4, 3), (5, 5))

    with Pool() as pool:
        res = pool.starmap(power, values)
        print(res)

    end = timer()
    print(f'elapsed time: {end - start}')


if __name__ == '__main__':
    main()
