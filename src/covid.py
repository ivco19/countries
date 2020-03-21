
import numpy as np
import pandas as pd
import datetime


# LOAD DATA
def load_data():
    base_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/' \
               + 'csse_covid_19_data/csse_covid_19_time_series/'

    # confirmed cases
    COVID_CONFIRMED_URL = base_url + 'time_series_19-covid-Confirmed.csv'
    covid_confirmed = pd.read_csv(COVID_CONFIRMED_URL)
    covid_confirmed_long = pd.melt(covid_confirmed,
                                   id_vars=covid_confirmed.iloc[:, :4],
                                   var_name='date',
                                   value_name='confirmed')

    # deaths
    COVID_DEATHS_URL = base_url + 'time_series_19-covid-Deaths.csv'
    covid_deaths = pd.read_csv(COVID_DEATHS_URL)
    covid_deaths_long = pd.melt(covid_deaths,
                                   id_vars=covid_deaths.iloc[:, :4],
                                   var_name='date',
                                   value_name='deaths')

    # recovered
    COVID_RECOVERED_URL = base_url + 'time_series_19-covid-Recovered.csv'
    covid_recovered = pd.read_csv(COVID_RECOVERED_URL)
    covid_recovered_long = pd.melt(covid_recovered,
                                   id_vars=covid_recovered.iloc[:, :4],
                                   var_name='date',
                                   value_name='recovered')

    covid_df = covid_confirmed_long
    covid_df['deaths'] = covid_deaths_long['deaths']
    covid_df['recovered'] = covid_recovered_long['recovered']
    covid_df['active'] = covid_df['confirmed'] - covid_df['deaths'] - covid_df['recovered']

    covid_df['Country/Region'].replace('Mainland China', 'China', inplace=True)
    covid_df[['Province/State']] = covid_df[['Province/State']].fillna('')
    covid_df.fillna(0, inplace=True)
    covid_df.drop(['Lat', 'Long'], axis=1, inplace=True)

    # WRITE DATA TO FILE
    covid_df.to_csv('../dat/covid_df.csv', index=None)

    # WRITE DATA BY COUNTRY
    covid_countries_df = covid_df.groupby(['Country/Region', 'Province/State']).max().reset_index()
    covid_countries_date_df = covid_df.groupby(['Country/Region', 'date'], sort=False).sum().reset_index()

    return(covid_countries_date_df)


def countries(df, country_name):

    # COUNTRIES
    df_arg = df[df['Country/Region'] == country_name]
    return(df_arg)


df_countries = load_data()


# PLOT ----------------------- casos confirmados en Argentina
def plt_1country(df, country_name):

    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(16, 6))

    sns.lineplot(x=df['date'], y=df['confirmed'], sort=False, linewidth=2)
    plt.suptitle("COVID-19", fontsize=16, fontweight='bold', color='white')

    plt.xticks(rotation=45)
    plt.ylabel('casos confirmados')

    ax.legend(['Argentina', 'World except China'])

    fplot = '../plt/plot_' + country_name + '.png'
    fig.savefig(fplot)


# ------------------




df_arg = countries(df_countries, 'Argentina')

plt_1country(df_arg, 'argentina')






