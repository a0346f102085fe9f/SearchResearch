import torch

# Weights for outer product
def weights(n):
	v = torch.arange(n)
	m = v - v.unsqueeze(1)
	w = 0.5**m

	return w.triu(1)

# Non-batch version
def score(arr):
	x = torch.tensor(arr).float()
	p = torch.outer(x, x).triu(1)
	b = weights(5) * p

	score = x.sum() + b.sum()

	return score

print(score([1, 0, 0, 0, 0]))
print(score([0, 0, 0, 0, 1]))
print(score([1, 0, 0, 0, 1]))
print(score([1, 1, 0, 0, 0]))
print(score([0, 1, 1, 0, 0]))
print(score([0, 0, 1, 1, 0]))
print(score([0, 0, 0, 1, 1]))
print(score([1, 0, 1, 0, 0]))
print(score([0, 1, 0, 1, 0]))
print(score([0, 0, 1, 0, 1]))
print(score([1, 0, 0, 1, 0]))
print(score([0, 1, 0, 0, 1]))
print(score([1, 1, 1, 1, 1]))
