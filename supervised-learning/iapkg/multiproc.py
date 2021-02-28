#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 07:54:07 2021

@author: Jeremy Levens
"""

import multiprocessing
import time

data = (
    ['a', '2'], ['b', '4'], ['c', '6'], ['d', '8'],
    ['e', '1'], ['f', '3'], ['g', '5'], ['h', '7']
)


def mp_worker(data_tuple):
    (inputs, the_time) = data_tuple
    print('Process %s\tWaiting %s seconds' % (inputs, the_time))
    time.sleep(int(the_time))
    print('Process %s\tDONE' % inputs)


def mp_handler():
    pool = multiprocessing.Pool(2)
    pool.map(mp_worker, data)


if __name__ == '__main__':
    mp_handler()
