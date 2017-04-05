#!/home/anilraj/anaconda3/bin/python

import os
import time
import subprocess
from itertools import product
from random import shuffle
from random import randint
from random import uniform

import shlex
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

def running_or_not():
  proc = subprocess.Popen("nvidia-smi", stdout=subprocess.PIPE)
  output = proc.stdout.read()
  output = "".join(map(chr, output))
  if output.find('No running processes found') == -1:
    return 1
  else:
    return 0

for x in combo:
  a = x[0]
  b = x[1]
  c = x[2]
  d = x[3]
  e = (int(randint(6, 80) / 16)) * 16
  f = (int(randint(16, 400) / 40)) * 40
  g = (int(randint(100, 800) / 200)) * 200
  h = (int(randint(84, 600) / 64)) * 64
  i = uniform(0.8, 1.0)
  j = uniform(0.8, 1.0)
  k = (int(randint(3, 10) / 2)) * 2
  l = (int(randint(8, 50) / 5)) * 5
  m = (int(randint(50, 100) / 25)) * 25
  n = (int(randint(42, 75) / 8)) * 8
  o = uniform(0.8, 1.0)
  p = uniform(0.8, 1.0)
  q = x[4]
  while(running_or_not()):
    time.sleep(1)
    print('-')
  cmd = "./Traffic_Sign_Classifier.v0.1.py -a %s -b %s -c %s -d %s -e %s -f %s -g %s -h %s -i %s -j %s -k %s -l %s -m %s -n %s -o %s -p %s -q %s" % (a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q)
  print(cmd)
  # proc = subprocess.Popen(cmd, shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
  subprocess.call(shlex.split(cmd), timeout=3600)
  time.sleep(20)

