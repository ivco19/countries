import numpy as np
from sys import argv
import sys

import cv19

#___________________________________
# Load settings

conf = cv19.parser()
conf.check_file(argv)
conf.read_config_file()
conf.load_filenames()
conf.load_parameters()

#___________________________________
# Prepare experiment

c = cv19.InfectionCurve()

## Modelo simple SIR:
#t, I, C, R = c.model_SIR(conf.p)
#lbls = ['I','C','R']
#c.plt_IC_n(t, [I, C, R], labels=lbls, fplot=conf.filenames.fname_sir)

## Modelo simple SEIR:
t, S, E, I, R = c.model_SEIR(conf.p)
lbls = ['S','E','I','R']
c.plt_IC_n(t, [S, E, I, R], labels=lbls, fplot=conf.filenames.fname_seir)

# Modelo simple SEIRF:
#t, S, E, I, R, F = c.model_SEIRF(conf.p)
#lbls = ['S','E','I','R','F']
#c.plt_IC_n(t, [S, E, I, R, F], labels=lbls, fplot=conf.filenames.fname_seir)
 
