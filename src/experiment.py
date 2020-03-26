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

c.compute_tst(conf.p)

#t, I, C, R = c.compute(conf.p)

#c.plt_IC_n(t, [I, C, R], conf.filenames.fname_infected)




# https://towardsdatascience.com/covid-19-infection-in-italy-mathematical-models-and-predictions-7784b4d7dd8d