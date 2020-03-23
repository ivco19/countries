import numpy as np
from sys import argv
import sys

import cv19

#___________________________________
# Load settings

from configparser import ConfigParser
filename = cv19.check_file(argv)
config = ConfigParser()
config.read(filename)

#___________________________________
# Prepare experiment


p = cv19.load_parameters(config._sections)

t, inf = cv19.InfectionCurve_full(p)
 
fn = cv19.load_filenames(config._sections)

cv19.plt_IC(t, inf, fn.fname_infected)





