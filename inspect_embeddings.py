from itertools import islice
from array import array
import random
import torch

from torch.nn import Linear

def load():
	src_file = open("datatape_src.bin", "rb") # Token frequency map
	dst_file = open("datatape_dst.bin", "rb") # Resulting embeddings

	src_read = lambda: array("i", src_file.read(32100*4))
	dst_read = lambda: array("f", dst_file.read(768*4))

	size = 8154

	src = [src_read() for _ in range(size)]
	dst = [dst_read() for _ in range(size)]

	random.seed(42)

	return random.sample(list(zip(src, dst)), size)

pairs = load()

class linear_model(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.linear1 = Linear(32100, 768)


	def forward(self, x):
		return self.linear1(x)

net = linear_model()
print(net)

# https://docs.python.org/3/library/itertools.html#itertools-recipes
def batched(iterable, n):
	it = iter(iterable)
	while (batch := tuple(islice(it, n))):
		yield batch

# MSE loss
loss_fn = lambda have, want: torch.mean(torch.square(have - want))

def test():
	src, dst = pairs[4548]

	src = torch.tensor(src).float()
	dst = torch.tensor(dst).float()

	loss = loss_fn(net(src), dst)
	print(loss)

# Batch size has a lot of effect
def train():
	for x in batched(pairs, 16):
		src = [torch.tensor(src) for src, _ in x]
		dst = [torch.tensor(dst) for _, dst in x]

		src = torch.vstack(src).float()
		dst = torch.vstack(dst)

		loss = loss_fn(net(src), dst)

		# Explodes if loss is uncapped
		if loss > 100:
			continue

		loss.backward()

		with torch.no_grad():
			net.linear1.weight -= net.linear1.weight.grad * 1e-5
			net.linear1.bias -= net.linear1.bias.grad * 1e-5
			net.linear1.weight.grad.zero_()
			net.linear1.bias.grad.zero_()

		#print(loss)
		test()

