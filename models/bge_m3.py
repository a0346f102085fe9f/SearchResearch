import torch
import huggingface_hub
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
model = AutoModel.from_pretrained("BAAI/bge-m3")

colbert_linear = torch.nn.Linear(1024, 1024)
sparse_linear = torch.nn.Linear(1024, 1)

colbert_linear.load_state_dict(torch.load(huggingface_hub.hf_hub_download("BAAI/bge-m3", "colbert_linear.pt")))
sparse_linear.load_state_dict(torch.load(huggingface_hub.hf_hub_download("BAAI/bge-m3", "sparse_linear.pt")))

# This mix of parameters allows for long inputs
# Resulting token_ids will look as follows:
# [[0, ..., 2],
#  [0, ..., 2],
#  [0, ..., 2],
#  [0, ..., 1]]
#
# 0 = [bos] / [cls]
# 2 = [eos]
# 1 = [pad]
#
# Similar to default BGE-M3 tokenization except overflowing tokens stay
#
# Notes:
# - A "stride" parameter is available to overlap slices by n tokens
# - A mapping of output rows to input is available, if array passed
def tokenize(x):
	return tokenizer(
		x,
		padding=True,
		truncation=True,
		return_overflowing_tokens=True,
		return_tensors='pt',
		max_length=8192,
	)

# We want to inspect what the model actually gets
inspect = lambda x: tokenizer.decode(x.input_ids.flatten().tolist())

def run_model(inputs):
	tokens = inputs.input_ids
	mask = inputs.attention_mask

	with torch.no_grad():
		result = model(tokens, attention_mask=mask).last_hidden_state

	# Dense embedding
	dense = torch.nn.functional.normalize(result[:, 0], dim=-1)

	# Sparse embedding
	sparse = torch.relu(sparse_linear(result))[:, :, 0]
	sparse = [ dict(zip(k, v)) for k, v in zip(tokens.tolist(), sparse.tolist()) ]
	# Optionally, use tokenizer.convert_ids_to_tokens(k)

	for dict_ in sparse:
		dict_.pop(0)
		dict_.pop(2)
		if 1 in dict_: dict_.pop(1)

	# Colbert embedding
	colbert = colbert_linear(result[:, 1:])
	colbert *= mask[:, 1:][:, :, None]
	colbert = torch.nn.functional.normalize(colbert, dim=-1)

	return dense.cpu(), sparse, colbert.cpu()

def run(x):
	assert type(x) is str

	inputs = tokenize(x)
	dense, sparse, colbert = run_model(inputs)

	return dense[0], sparse[0], colbert[0]

sparse_similarity = lambda a, b: sum([a[k] * b[k] for k in set(a).intersection(set(b))])

def similarity(a, b):
	dense_a, sparse_a, colbert_a = a
	dense_b, sparse_b, colbert_b = b

	dense_score = torch.dot(dense_a, dense_b)
	sparse_score = sparse_similarity(sparse_a, sparse_b)

	return (4/6)*dense_score + (2/6)*sparse_score

# Reference implementation:
# https://github.com/FlagOpen/FlagEmbedding
#
# It can be used without installation
#
def selftest():
	def test_embeddings():
		sentences = ["What is BGE M3?", "Defination of BM25"]

		inputs = tokenize(sentences)
		dense, sparse, colbert = run_model(inputs)

		reference_dense = torch.tensor(
			[[-0.03406, -0.047, -0.000947],
			 [-0.01043, -0.04483, -0.02434]]
		)

		# We don't want to check relative tolerance... set it to 1
		assert torch.all(torch.isclose(dense[:, :3], reference_dense, rtol=1, atol=0.001))

		reference_sparse_1 = {
			4865: 0.0837,
			83: 0.08136,
			335: 0.1295,
			11679: 0.2517,
			276: 0.17,
			363: 0.2695,
			32: 0.0408
		}

		reference_sparse_2 = {
			262: 0.0501,
			5983: 0.137,
			2320: 0.04517,
			111: 0.06335,
			90017: 0.2517,
			2588: 0.3333
		}

		def dict_close(a, b, eps=0.001):
			if a.keys() != b.keys():
				return False

			for m, n in zip(a, b):
				if m - n >= eps:
					return False

			return True

		assert dict_close(reference_sparse_1, sparse[0])
		assert dict_close(reference_sparse_2, sparse[1])

	def test_scores():
		sentences_1 = ["What is BGE M3?", "Defination of BM25"]
		sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

		queries = tokenize(sentences_1)
		documents = tokenize(sentences_2)

		dense_q, sparse_q, colber_q = run_model(queries)
		dense_d, sparse_d, colber_d = run_model(documents)

		similarity_dense = dense_q @ dense_d.T

		reference_similarity_dense = torch.tensor([
			[0.6265, 0.3477],
			[0.3499, 0.678]
		])

		assert torch.all(torch.isclose(similarity_dense, reference_similarity_dense, rtol=1, atol=0.001))

		assert sparse_similarity(sparse_q[0], sparse_d[0]) - 0.19554901123046875 < 0.01
		assert sparse_similarity(sparse_q[0], sparse_d[1]) - 0.0 < 0.01

	test_embeddings()
	test_scores()

#selftest()
