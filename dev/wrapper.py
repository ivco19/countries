# cargar herramientas de arcovid ----------------
from arcovid19.models import Graph
from arcovid19.models import InfectionCurve as IC

# cargar configuracion --------------------------
# (esto esta hardcodeado, pero iria de otra forma, levantando
# el archivo de configuracion o los parametros de la página... no sé)

N_init = 5
population = 1.e6
incubation = 5
dt = 1

"""
aca tambien hay que levantar las tablitas de probabilidades
que hicieron Dante y Fede, o tablas de la literatura, o tablas
propuestas como modelo...  Despues se usan para cargar las aristas.
"""

T1 = [random() for _ in range(Nbins)]  # <-- esto tiene que ser una PDF
T2 = [random() for _ in range(Nbins)]  # <-- esto tiene que ser una PDF
# etc.

# crear y completar el grafo ----------------------

IDM = arcovid19.IDModel()  # Infectious desease model

for compartment in ['S','E','I','R']:
    IDM.add_compartment(node, 0)

IDM.set_compartment('S', population)
IDM.set_compartment('E', 0)
IDM.set_compartment('I', N_init)
IDM.set_compartment('R', 0)

nms = ['prob','lag']
g.add_transition('S', 'S', nms, [T1, T2])
g.add_transition('E', 'E', nms, [T1, T2])
g.add_transition('I', 'I', nms, [T1, T2])

g.add_transition('S', 'E', nms, [T1, T2])
g.add_transition('E', 'I', nms, [T1, T2]) #[, tiempo de incubacion]
g.add_transition('I', 'R', nms, [T1, T2]) #[, tiempo de recuperacion]

# Para pasar de S y E a I, hacer:
g.add_transition(['S', 'E'], 'I', nms, [T1, T2])



# aca son distintos "T1" y "T2" para cada edge.

# correr el modelo -----------------------------------

# parametros incluye:
#    tiempo de corrida de la simulacion
#    valores iniciales que hagan falta
#    en general... todo lo que haga falta y no este en el grafo

arcovid19.models.run(g, parametros)

donde...

def arcovid19_models_run(g, p):
    """Ejecuta la simualacion para un modelo (g, p)

    g es el grafo y p son los parametros.  El grafo tiene un nodo
    por cada compartimiento, y contiene valores separados por edad.
    Las transiciones son probabilisticas, y van en las "aristas".
    """
    ts = [0.] # time series
    p_dt = 1.
    T_IC = int(incubation/ dt)

    t = 0.
    time_steps = 0

    while t < parametros['t_max']:
        time_steps = time_steps + dt
        t_prev = t
        t = t + p.dt
        ts.append(t)

        act = []
        for N in g:
            act.append(N.node_activation())

        for i, N in enumerate(g):
            g.node_upgrade(act[i])
