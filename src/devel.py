Hola Juan, 

bueno, te cuento la idea de la implementacion de los modelos, 
falta darle cierre y necesito un poco de feedback.

La idea original, que motivó todo el kilombo de los grafos, era hacer
cualquier modelo y guardarlo como un grafo.  Cada nodo representa un 
compartimiento (que puede tener un valor escalar, o una lista que 
representa los etarios), y cada arista contiene las probabilidades 
de transisón y las probabilidades de retardo, en forma tambien de
listas (o arrays... lo importante es que pueda tener varios valores).

Una vez cargado el grafo con el modelo, se inicia con una determinada
poblacion y se hace evolucionar en una especie de "simulación", a la
que se le puede agregar alguna componente estocástica para correrlo
muchas veces y hacer una estimacion MonteCarlo de la incerteza de cada
cosa en funcion del tiempo.

Entonces la historia sería mas o menos asi:



#_____________________________________
# cargar herramientas de arcovid
from arcovid19.models import Graph
from arcovid19.models import InfectionCurve as IC


#_____________________________________
# cargar configuracion
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


#_____________________________________
# crear y completar el grafo
g = Graph()
for node in ['S','E','I','R']:
    g.add_node(node, 0)
    
g.set_node('S', population)
g.set_node('E', 0)
g.set_node('I', N_init)
g.set_node('R', 0)

nms = ['prob','lag']
g.add_edge('S', 'S', nms, [T1, T2])
g.add_edge('E', 'E', nms, [T1, T2])
g.add_edge('I', 'I', nms, [T1, T2])

g.add_edge('S', 'E', nms, [T1, T2])
g.add_edge('E', 'I', nms, [T1, T2]) #[, tiempo de incubacion]
g.add_edge('I', 'R', nms, [T1, T2]) #[, tiempo de recuperacion]

# aca son distintos "T1" y "T2" para cada edge.

#_____________________________________
# correr el modelo

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
    
        time_steps = time_steps + 1
    
        t_prev = t
        t = t + p.dt
        ts.append(t)

        act = []
        for N in g:
            act.append(N.node_activation())

        for i, N in enumerate(g):
            g.node_upgrade(act[i])



o sea, esa funcion lo que hace es:

   - recorre todos los nodos
   - por cada nodo busca cuales son los nodos conectados que le
     apuntan (las aristas tienen direccion)
   - calcula la "activacion" (si, ya se...)
   - guarda las variaciones en cada uno de los nodos
   - cuando termina, incorpora esas variaciones a todos
     los nodos.  No se puede puede ir haciendo sobre la marcha
     porque dependeria del orden de la actualizacion.

Esta es una forma muy chota de resolver una ecuacion diferencial, pero
dada la incerteza en los parámetros del modelo me parece que no vale
la pena delirarse con la precisión del vigésimo dígito.



Los modelos que ya están (SIR, SEIR, SEIRF), 
tienen básicamente las funciones de los nodos
desparramadas.  O sea, cuando lo hice estaba pensando en la estructura
de la funcion de arriba, porque como veras esta hecho de tal forma que 
see repite la misma estructura en cada compartimiento.


Bueno, decime que te parece la estructura y armo una version que se
pueda correr.




