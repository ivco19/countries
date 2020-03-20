# covid19_iate

Tools to analize covid19 data

Resources
---------

Data sources:


- `Clean dat from OpenBlender  <https://towardsdatascience.com/gather-all-the-coronavirus-data-with-python-19aa22167dea>`_
- `Analyzing Coronavirus (Covid-19) Data Using Pandas and Plotly <https://towardsdatascience.com/analyzing-coronavirus-covid-19-data-using-pandas-and-plotly-2e34fe2c4edc>`_
- `Notebook in RMOTR <https://notebooks.ai/rmotr-curriculum/covid-19-40c03c06>`_
- `MathWorks blog <https://blogs.mathworks.com/loren/2020/03/16/analyzing-novel-corona-virus-covid-19-dataset/>`_

Pandemic:

- `COVID-19 Daily Tracker <https://rpubs.com/thelilster/583398>`_
- `The handbook project <https://coronavirustechhandbook.com/>`_
- `Article in KDnuggets <https://www.kdnuggets.com/2020/03/covid-19-your-community-you-data-science-perspective.html>`_
- `Infection Trajectory in VisualCapitalist <https://www.visualcapitalist.com/infection-trajectory-flattening-the-covid19-curve/>`_

Notebooks
---------

Hasta ahora hay dos notebooks que son mas que nada exploratorios de los datos compilados por `Johns Hopkins CSSE<https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30120-1/fulltext>`_

world.ipynb:
   carga datos de la poblacion y area de los paises para construir una tabla con esos dos datos.  Luego se usará para normalizar las curvas de contagio de los diferentes países.

load_data.ipynb:
   Carga los datos de JHU CSSE y analiza las curvas de contagio de algunos países.



DATA
---------

Los datos actualizados son leidos directamente de GitHub, no hace falta bajarlos.

Los datos sobre poblacion y area de los países:

table-1.csv:
   source: `<https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_area>`_
   preprocessed with `<https://wikitable2csv.ggor.de/>`_

table-2.csv:
   source: `<https://en.wikipedia.org/wiki/List_of_countries_by_population_(United_Nations)>`_
   preprocessed with `<https://wikitable2csv.ggor.de/>`_

world_area.csv:
   Tabla limpia con las areas

world_population.csv:
   Tabla limpia con las poblaciones

world.ods:
   Archivo ODS con las dos tablas (para verificar a ojo)

pop_area.csv:
   Tabla con las columnas de poblacion y area combinadas















