import torch

m = 8
n = 4
k = 8

a = torch.tensor(list(range(0, m * k))).reshape(m, k)
b = torch.tensor(list(range(0, n * k))).reshape(n, k)

print(a)
print(b)
print(torch.matmul(a, b.t()))
