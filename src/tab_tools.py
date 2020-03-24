#class tabular

import pandas as pd
from random import random

class table_draw:

    def __init__(self):
        self.size = []
        self.ranges = []
        self.prob = []
        self.dif_prob = []

    def load(self, filename):
        D = pd.read_csv(filename)
        self.D = D
        self.size = D.shape[0]

    def add_clean_column(self, columnname, column):
        self.D[columnname] = column

    def tabular(self, x, y):
        self.x = self.d[x]
        self.p = self.d[y]
        
    def random_gen(self, Xname, Yname):
        u = random()
        j = self.size
        for i, y in enumerate(self.D[Yname]):
            if u > y:
                j = i

        print(self.D['x'][j])






if __name__ == '__main__':

    T = table_draw()

    filename = '../dat/table_population_cordoba.csv'

    T.load(filename)

    l = []
    for s in T.D['age_range']:
        i = s.split('-')[1]
        l.append(int(i))
    T.add_clean_column('x',l)

    l = []
    for s in T.D['acum_F']:
        l.append(float(s)/100.)

    T.add_clean_column('y',l)

    T.random_gen('x','y')






