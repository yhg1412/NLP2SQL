import torch.nn as nn
import torch
import numpy
import datetime

a = torch.randn((1, 300))
b = torch.randn((300, 1))

N = 300000
len = 100000
step = len/20

c = torch.randn(N)
d = torch.randn(N/2)
L = nn.Linear(N, 1)

print(datetime.datetime.now())
for i in range(len):
    b = L(c)
    if i % step == 0:
        print(i)
print(datetime.datetime.now())
for i in range(len):
    b = torch.dot(d, d)
    if i % step == 0:
        print(i)
print(datetime.datetime.now())