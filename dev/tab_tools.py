#class tabular

import pandas as pd
from random import random

class table_draw:
    #{{{

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
        
    def random_gen(self, xinfname, xsupname, yname):

        import sys
        u = random()
        j = self.size

        #msg = 'testing if %f is in interval (%f %f) / index: %i'
        y_old = 0.
        for i, y_new in enumerate(self.D[yname]):
            if u > y_old and u < y_new:
                j = i
                #print(msg % (u, y_old, y_new, i))
                break
            else:
                y_old = y_new

        if j<0 or j>= self.size: 
            print(j, self.size)
            sys.exit()

        x1 = self.D[xinfname][j]
        x2 = self.D[xsupname][j]
        res = (u-y_old)/(y_new-y_old)*(x2-x1) + x1
        res = int(res)
        return(res)

    def test_random_gen(self, nran, fplot):

        r = []
        for _ in range(nran):
            r.append(self.random_gen('x_inf', 'x_sup','y'))

        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 10))
       
        plt.hist(r)
        plt.suptitle("random gen test", fontsize=16, \
                fontweight='bold', color='white')
        plt.xticks(rotation=0)
        plt.xlabel('X')
        plt.ylabel('PDF')
       
        fig.savefig(fplot)
        plt.close()
       
    def save_clean_CDF(self, filename):

        df = self.D[['x_inf', 'x_sup','y']]
        df.to_csv(filename)

    #}}}

