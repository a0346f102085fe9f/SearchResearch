import torch

# Weights for outer product
def weights(n):
	v = torch.arange(n)
	m = v - v.unsqueeze(1)
	w = 0.5**m

	return w.triu(1)

# Proximity scoring
# Relies on internal batching to deal with subpar memory scaling
# Before: 4*30522*512*512 bytes = 29.8 GB
# After: 4*128*512*512 bytes = 128 MB
def score(x):
	w = weights(x.shape[1])
	slices = x.split(128)
	sums = []

	for s in slices:
		p = torch.einsum("bi,bj->bij", s, s).mul_(w)
		sums.append( p.sum((1,2)) )

	score = x.sum(1) + torch.cat(sums)

	return score


A = torch.zeros(30522, 8)
A[0, 0] = 1.0

B = torch.zeros(30522, 8)
B[0, -1] = 1.0

C = torch.zeros(30522, 8)
C[0, 0] = 1.0
C[0, 1] = 1.0

D = torch.zeros(30522, 8)
D[0, 0] = 1.0
D[0, -1] = 1.0

E = torch.zeros(30522, 512)
E[0, 0] = 1.0
E[0, -1] = 1.0

print(score(A))
print(score(B))
print(score(C))
print(score(D))
print(score(E))
