import numpy as np
import torch
import matchzoo as mz
a = np.array(torch.LongTensor([1,0,1,0,1]))
b = np.array(torch.LongTensor([0,1,1,1,0]))
print(np.logical_and((a==b),b))
a = torch.LongTensor([1,0,1,0,1])
print(torch.logical_and(a,a))
print(mz.models.BiMPM)
def shabi():
    pass
for i in range(10):
    print(1)
