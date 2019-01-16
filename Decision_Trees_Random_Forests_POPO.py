import numpy as np
import pandas as pd
from sklearn import tree

input_file = "e:/sundog-consult/udemy/datascience/PastHires.csv"
df = pd.read_csv(input_file, header = 0)
df.head()

d = {'Y': 1, 'N': 0}
df['Hired'] = df['Hired'].map(d)
df['Employed?'] = df['Employed?'].map(d)
df['Top-tier school'] = df['Top-tier school'].map(d)
df['Interned'] = df['Interned'].map(d)
d = {'BS': 0, 'MS': 1, 'PhD': 2}
df['Level of Education'] = df['Level of Education'].map(d)
df.head()


features = list(df.columns[:6])
print(features)