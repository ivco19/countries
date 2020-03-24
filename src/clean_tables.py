import tab_tools as tt


T = tt.table_draw()


# POBLACION DE CORDOBA ::::::::::::::::::::::::::::::::::::::

filename = '../dat/table_population_cordoba.csv'

T.load(filename)

l1 = []
l2 = []
for s in T.D['age_range']:
    i = s.split('-')
    l1.append(int(i[0]))
    l2.append(int(i[1]))
T.add_clean_column('x_inf',l1)
T.add_clean_column('x_sup',l2)

l = []
for s in T.D['acum_F']:
    l.append(float(s)/100.)

T.add_clean_column('y',l)
T.test_random_gen(1000, '../plt/test_randoms_001.png')

T.save_clean_CDF('../dat/table_clean.csv')

 
