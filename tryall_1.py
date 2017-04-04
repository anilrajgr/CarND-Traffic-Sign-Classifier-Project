#!/home/anilraj/anaconda3/bin/python

import os
import time
from subprocess import Popen
from itertools import product
from random import shuffle
from random import randint
from random import uniform

import shlex
from subprocess import call
# print(shlex.split(cmd))

all_epochs = {10, 15, 20, 50, 100, 500}
all_batchsize = {64, 128, 256, 512} # , 1024, 2048}
all_mu = {0}
all_sigma = {0.1, 0.01, 0.001}
# all_conv1depth = {6, 10, 20, 60, 200}
# all_conv2depth = {16, 40, 100, 500}
# all_fc1size = {100, 200, 400, 1000}
# all_fc2size = {84, 116, 200, 500}
# all_conv1keepprob = {0.5, 0.8, 0.9, 0.95, 1.0}
# all_conv2keepprob = {0.5, 0.8, 0.9, 0.95, 1.0}
# all_conv1adepth = {3, 5, 10, 30, 100}
# all_conv2adepth = {8, 20, 50, 250}
# all_fc1asize = {50, 100, 200, 500}
# all_fc2asize = {42, 58, 100, 250}
# all_conv1akeepprob = {0.5, 0.8, 0.9, 0.95, 1.0}
# all_conv2akeepprob = {0.5, 0.8, 0.9, 0.95, 1.0}
all_learningrate = {0.001} # {0.01, 0.001, 0.0001} Other learning rates didn't work well.

numcores = ncpus = os.sysconf("SC_NPROCESSORS_ONLN")

combo = list(product(all_epochs, all_batchsize, all_mu, all_sigma, all_learningrate))
shuffle(combo)

def busy_or_not():
  if (os.getloadavg()[0] / numcores > 0.9):
    return 1
  else:
    return 0

for x in combo:
  a = x[0]
  b = x[1]
  c = x[2]
  d = x[3]
  e = randint(6, 20)
  f = randint(16, 100)
  g = randint(100, 200)
  h = randint(84, 150)
  i = uniform(0.8, 1.0)
  j = uniform(0.8, 1.0)
  k = randint(3, 10)
  l = randint(8, 50)
  m = randint(50, 100)
  n = randint(42, 75)
  o = uniform(0.8, 1.0)
  p = uniform(0.8, 1.0)
  q = x[4]
  while(busy_or_not()):
    time.sleep(10)
  cmd = "./Traffic_Sign_Classifier.v0.1.py -a %s -b %s -c %s -d %s -e %s -f %s -g %s -h %s -i %s -j %s -k %s -l %s -m %s -n %s -o %s -p %s -q %s" % (a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q)
  # proc = Popen(cmd, shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
  call(shlex.split(cmd), timeout=3600)
  time.sleep(5)

