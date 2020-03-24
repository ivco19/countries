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

t, I = c.compute(conf.p)

c.plt_IC_n(t, [I], conf.filenames.fname_infected)



