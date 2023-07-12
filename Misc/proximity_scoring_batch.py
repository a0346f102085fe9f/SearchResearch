import torch

# Weights for outer product
def weights(n):
	v = torch.arange(n)
	m = v - v.unsqueeze(1)
	w = 0.5**m

	return w.triu(1)

# Proximity scoring
def score(x):
	w = weights(x.shape[1]).flatten()
	sums = []

	for s in x:
		sum = torch.outer(s, s).flatten().dot(w)
		sums.append(sum)

	score = x.sum(1) + torch.tensor(sums)

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
