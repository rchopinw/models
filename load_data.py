import numpy as np
import pandas as pd
from collections import Counter, deque
from itertools import chain, product
import heapq
from io import StringIO


# using with statement to open a file
# open function:
# r: open for reading (default)
# w: open for writing, truncating the file first
# a: open for writing, appending to the end of the file if it exists
# b: open in binary mode
# t: text mode
# +: open for updating, both reading and writing
file_path = 'try.txt'
content = []
with open(file_path, ) as fp:
    lines = fp.readlines()
    for line in lines:
        content.append(line)


# using numpy to open a file
np.loadtxt('try.txt',
           delimiter=',',
           dtype={'names': ('gender', 'age', 'weight'),
                  'formats': ('S1', 'i4', 'f4')})


# using pandas
data = pd.read_csv('https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt', sep='\t')
df = pd.DataFrame([[10, 30, 40], [], [15, 8, 12],
                   [15, 14, 1, 8], [7, 8], [5, 4, 1]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3', 'Basket4',
                         'Basket5', 'Basket6'])
print(df.unstack(level=-1))