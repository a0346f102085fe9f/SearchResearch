from huggingface_hub import hf_hub_download
from transformers import T5EncoderModel
import torch.nn.functional as F
import torch

# Simplified Mean Pooling
# https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Pooling.py
class Pooling(torch.nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, token_embeddings, attention_mask):
		input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
		sum = torch.sum(token_embeddings * input_mask_expanded, 1)

		# All zeros attention mask will cause breakage
		# So avoid doing that
		n = input_mask_expanded.sum(1)

		return sum / n

# Dense layer
# https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Dense.py
class Dense(torch.nn.Module):
	def __init__(self, path):
		super().__init__()

		dense_weights = torch.load(path, map_location="cpu")
		self.linear = torch.nn.Linear(768, 768, bias=False)
		self.linear.weight = torch.nn.Parameter(dense_weights["linear.weight"])
		self.activation_function = torch.nn.modules.linear.Identity()

	def forward(self, sentence_embedding):
		return self.activation_function(self.linear(sentence_embedding))

# Normalization
# https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Normalize.py
class Normalize(torch.nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, sentence_embedding):
		return F.normalize(sentence_embedding, p=2, dim=1)

# Improved INSTRUCTOR class
# Does not return a dict
class INSTRUCTOR(torch.nn.Module):
	def __init__(self):
		super().__init__()

		# Base
		# https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Transformer.py
		self.base = T5EncoderModel.from_pretrained("hkunlp/instructor-base")
		self.pooling = Pooling()
		self.dense = Dense(hf_hub_download(repo_id="hkunlp/instructor-base", filename="2_Dense/pytorch_model.bin"))
		self.normalize = Normalize()

	def forward(self, input):
		# Stage 0: Base model
		# Attention mask omitted
		# Equivalent to all ones attention mask
		token_embeddings = self.base(input_ids = input["input_ids"], return_dict=False)[0]

		# Stage 1: Pooling
		sentence_embedding = self.pooling(token_embeddings, input["attention_mask"])

		# Stage 2: Dense layer
		sentence_embedding = self.dense(sentence_embedding)

		# Stage 3: Normalize
		sentence_embedding = self.normalize(sentence_embedding)

		return sentence_embedding
