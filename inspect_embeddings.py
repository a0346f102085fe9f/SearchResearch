from itertools import islice
from array import array
import random
import torch

from torch.nn import Linear

def load():
	src_file = open("datatape_src.bin", "rb") # Token frequency map
	dst_file = open("datatape_dst.bin", "rb") # Resulting embeddings

	src = torch.frombuffer(src_file.read(), dtype=torch.int32).split(32100)
	dst = torch.frombuffer(dst_file.read(), dtype=torch.float32).split(768)

	size = 8154

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

	src = src.float()
	dst = dst.float()

	loss = loss_fn(net(src), dst)
	print(loss)

# Batch size has a lot of effect
def train():
	for x in batched(pairs, 16):
		src = [src for src, _ in x]
		dst = [dst for _, dst in x]

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

