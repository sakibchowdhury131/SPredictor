from get_data import get_data
import torch
from StockCore import StockPred
import torch.nn as nn


data = torch.FloatTensor(get_data())[:, :128].unsqueeze(0)
stockPred = StockPred()
print(data.shape)
print(stockPred(data).shape)


'''
# With square kernels and equal stride
m = nn.Conv2d(16, 33, 3, stride=2)
# non-square kernels and unequal stride and with padding
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
# non-square kernels and unequal stride and with padding and dilation
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
input = torch.randn(20, 16, 50, 100)
output = m(input)
print(output.shape)
'''