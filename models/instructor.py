import torch
from transformers import AutoModel, AutoTokenizer
from models.instructor_class import INSTRUCTOR

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Notes about INSTRUCTOR:
# - Instructions and the text are joined without adding a space
tokenizer = AutoTokenizer.from_pretrained("hkunlp/instructor-base")
model = INSTRUCTOR()
model.eval()
model.to(device)

# Fixed instructions
instructions = tokenizer("Represent the document for retrieval:", return_tensors='pt', add_special_tokens=False)
instructions_size = instructions['attention_mask'].sum(1)

# Cosine similarity recommended for this model
mag = lambda x: x.dot(x)**0.5
similarity = lambda a, b: (a @ b) / (mag(a) * mag(b))

# This mix of parameters allows for long inputs and leaves space for instructions
# Every slice is prefixed with instructions
# Resulting token_ids will look as follows:
# [[instructions, ..., 1],
#  [instructions, ..., 1],
#  [instructions, ..., 1],
#  [instructions, ..., 0]]
#
# A "stride" parameter is available to overlap slices by n tokens
def tokenize(x):
	doc = tokenizer(x, truncation=True, padding=True, return_overflowing_tokens=True, return_tensors="pt", max_length=512-instructions_size)

	a = doc['input_ids']
	b = doc['attention_mask']

	vsz = len(a)

	ia = instructions['input_ids']
	ib = instructions['attention_mask']*0

	iadm = ia.repeat([vsz, 1])
	ibdm = ib.repeat([vsz, 1])

	return {
		'input_ids': torch.cat([iadm, a], dim=1),
		'attention_mask': torch.cat([ibdm, b], dim=1),
	}


# We want to inspect what the model actually gets
inspect = lambda x: tokenizer.decode(x['input_ids'].flatten().tolist())

# Long inputs pooled using sum(0)
# Alternative: flatten()
pool = lambda x: x.sum(0)

# Max batch size
bsz = 4

def run_model(inputs):
	with torch.no_grad():
		emb = model(inputs)

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
			'attention_mask': b.to(device),
		}

		slices.append( run_model(inputs) )

	return pool( torch.vstack(slices) )

# Sanity checking
def selftest():
	reference = tokenizer(["Represent the Science title:3D ActionSLAM: wearable person tracking in multi-floor environments"], return_tensors='pt').to(device)
	reference['attention_mask'][:, :6] = 0
	#reference['context_masks'] = torch.tensor([6]).to(device)
	embedding = run_model(reference)

	assert embedding[0][-1] == 0.015499825589358807
	assert embedding[0][-2] == -0.023956988006830215
	assert embedding[0][-3] == -0.006982537917792797
	assert embedding[0][-4] == 0.019618984311819077

#selftest()
