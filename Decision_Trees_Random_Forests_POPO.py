# import numpy as np
import pandas as pd
# from sklearn import tree

import csv

input_file = "/Users/CommanderCarr/Coding/python/data_science/DataScience-Python/PastHires.csv"
df = pd.read_csv(input_file, header = 0)
# print(df.head())

# with open("/Users/CommanderCarr/Coding/python/data_science/DataScience-Python/PastHires.csv") as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     line_count = 0
#     for row in csv_reader:
#         if line_count == 0:
#             print(f'Column names are {", ".join(row)}')
#             line_count += 1
#         else:
#             print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
#             line_count += 1
#     print(f'Processed {line_count} lines.')



d = {'Y': 1, 'N': 0}
df['Hired'] = df['Hired'].map(d)
df['Employed?'] = df['Employed?'].map(d)
df['Top-tier school'] = df['Top-tier school'].map(d)
df['Interned'] = df['Interned'].map(d)
d = {'BS': 0, 'MS': 1, 'PhD': 2}
df['Level of Education'] = df['Level of Education'].map(d)
print(df.head())
#
#
# features = list(df.columns[:6])
# print(features)