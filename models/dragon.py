import torch
from transformers import AutoModel, AutoTokenizer

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Facebook's DRAGON+ comes in two variants: document encoder and query encoder
tokenizer = AutoTokenizer.from_pretrained("facebook/dragon-plus-context-encoder")
model = AutoModel.from_pretrained("facebook/dragon-plus-context-encoder")
model.eval()
model.to(device)

# Dot product recommended for this model
similarity = lambda a, b: a @ b

# This mix of parameters allows for long inputs
# Resulting token_ids will look as follows:
# [[101, ..., 102],
#  [101, ..., 102],
#  [101, ..., 102],
#  [101, ..., 0]]
#
# A "stride" parameter is available to overlap slices by n tokens
tokenize = lambda x: tokenizer(x, max_length=512, truncation=True, padding=True, return_overflowing_tokens=True, return_tensors="pt")

# We want to inspect what the model actually gets
inspect = lambda x: [tokenizer.decode(encoding.ids) for encoding in x.encodings]

# Long inputs pooled using mean(0)
# Alternative: flatten()
pool = lambda x: x.mean(0)

# Max batch size
bsz = 4

def run_model(inputs):
	with torch.no_grad():
		emb = model(**inputs).last_hidden_state[:, 0, :]

	return emb.cpu()

def run(inputs):
	inputs = tokenize(inputs)

	# Apply batch size limit
	a_batch = inputs['input_ids'].split(bsz)
	b_batch = inputs['attention_mask'].split(bsz)

	slices = []

	for a, b in zip(a_batch, b_batch):
		inputs = {
			'input_ids': a.to(device),
			'attention_mask': b.to(device)
		}

		slices.append( run_model(inputs) )

	return pool( torch.vstack(slices) )
