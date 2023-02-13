from torch.nn import Linear
import torch

torch.manual_seed(42)

def load():
	src_file = open("datatape_src.bin", "rb") # Token frequency map
	dst_file = open("datatape_dst.bin", "rb") # Resulting embeddings

	size = 8154

	src = torch.frombuffer(src_file.read(), dtype=torch.int32).view(size, 32100)
	dst = torch.frombuffer(dst_file.read(), dtype=torch.float32).view(size, 768)

	return src, dst

xsrc, xdst = load()

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
	src = xsrc[4548]
	dst = xdst[4548]

	src = src.float()
	dst = dst.float()

	loss = loss_fn(net(src), dst)
	print(loss)

# Batch size has a lot of effect
bsz = 16

def train():
	batches = torch.randperm(8154).split(bsz)

	for indices in batches:
		src = xsrc[indices].float()
		dst = xdst[indices]

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


