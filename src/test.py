import numpy as np
import torch

a = np.array(torch.LongTensor([1,2,3]))
b = np.array(torch.LongTensor([1,2,3]))
print(np.sum(a==1))