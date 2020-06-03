import pandas as pd


df = pd.read_csv('CFR_WorldData.csv')

"""
Pasar de:
    1. Entity
    2. Code
    3. Date
    4. Total confirmed deaths due to COVID-19 (deaths)
    5. Total confirmed cases of COVID-19 (cases)

a las siguientes tablas, una por pais
    1. date
    2. total cases
    3. total deaths
"""

filt = df['code']=='AFG'

d = df[filt]

tot = []
cfr = []
startday = dt.datetime(year=2020, month=1, day=1)
for i in range(len(d)):
    date = dt.datetime.strptime(d['date'][i], '%b %d, %Y')
    #tot.append(d['cases'][i])
    #cfr.append(d['deaths'][i]) / max(1, d['cases'][i])


