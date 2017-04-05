#!/home/anilraj/anaconda3/bin/python

import os
import time
import subprocess
from itertools import product
from random import shuffle

import shlex
# print(shlex.split(cmd))

all_epochs = {10, 15, 20, 50} # , 100, 500}
all_batchsize = {64, 128, 256, 512} # , 1024, 2048}
all_mu = {0}
all_sigma = {0.1, 0.01, 0.001}
all_conv1depth = {20, 60, 200}
all_conv2depth = {100, 500}
all_conv3depth = {100, 500}
all_fc1size = {100, 200, 400, 1000}
all_fc2size = {84, 116, 200, 500}
all_conv1keepprob = {0.8, 0.9, 0.95, 1.0}
all_conv2keepprob = {0.8, 0.9, 0.95, 1.0}
all_conv3keepprob = {0.8, 0.9, 0.95, 1.0}
all_learningrate = {0.001} # {0.01, 0.001, 0.0001} Other learning rates didn't work well.

numcores = ncpus = os.sysconf("SC_NPROCESSORS_ONLN")

combo = list(product(all_epochs, all_batchsize, all_mu, all_sigma, all_conv1depth, all_conv2depth, all_conv3depth, all_fc1size, all_fc2size, all_conv1keepprob, all_conv2keepprob, all_conv3keepprob, all_learningrate))
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
  e = x[4]
  f = x[5]
  g = x[6]
  h = x[7]
  i = x[8]
  j = x[9]
  k = x[10]
  l = x[11]
  m = x[12]
  while(running_or_not()):
    time.sleep(1)
    print('-')
  cmd = "./Traffic_Sign_Classifier.v0.3.py -a %s -b %s -c %s -d %s -e %s -f %s -g %s -h %s -i %s -j %s -k %s -l %s -m %s" % (a, b, c, d, e, f, g, h, i, j, k, l, m)
  print(cmd)
  # proc = subprocess.Popen(cmd, shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
  subprocess.call(shlex.split(cmd), timeout=3600)
  time.sleep(20)

