import torch

# Weights for outer product
def weights(n):
	v = torch.arange(n)
	m = v - v.unsqueeze(1)
	w = 0.5**m

	return w.triu(1)

def score(x):
	p = torch.einsum("bi,bj->bij", x, x).triu(1)
	b = weights(p.shape[1]) * p

	score = x.sum(1) + b.sum((1,2))

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

print(score(A))
print(score(B))
print(score(C))
print(score(D))
