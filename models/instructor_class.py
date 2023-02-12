from sentence_transformers.models import Transformer, Pooling, Dense, Normalize
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig
from collections import OrderedDict
import json
import os

class INSTRUCTOR_Transformer(Transformer):
	def forward(self, features):

		# Attention mask omitted for base model
		# Equivalent to all ones attention mask
		trans_features = {
			'input_ids': features['input_ids']
		}

		output_states = self.auto_model(**trans_features, return_dict=False)
		output_tokens = output_states[0]

		features['token_embeddings'] = output_tokens

		return features

	@staticmethod
	def load(input_path: str):
		#Old classes used other config names than 'sentence_bert_config.json'
		for config_name in ['sentence_bert_config.json', 'sentence_roberta_config.json', 'sentence_distilbert_config.json', 'sentence_camembert_config.json', 'sentence_albert_config.json', 'sentence_xlm-roberta_config.json', 'sentence_xlnet_config.json']:
			sbert_config_path = os.path.join(input_path, config_name)
			if os.path.exists(sbert_config_path):
				break

		with open(sbert_config_path) as fIn:
			config = json.load(fIn)
		return INSTRUCTOR_Transformer(model_name_or_path=input_path, **config)

class INSTRUCTOR(SentenceTransformer):
	def _load_sbert_model(self, model_path):
		"""
		Loads a full sentence-transformers model
		"""

		# INSTRUCTOR uses a customized Transformer module
		# Also adds some additional blocks after the transformer
		# Simplified to remove dependency on importlib
		# https://huggingface.co/hkunlp/instructor-base/blob/main/modules.json
		# https://huggingface.co/hkunlp/instructor-base/blob/main/1_Pooling/config.json
		modules = OrderedDict()

		modules['0'] = INSTRUCTOR_Transformer.load(model_path)
		modules['1'] = Pooling(768)
		modules['2'] = Dense.load(os.path.join(model_path, "2_Dense"))
		modules['3'] = Normalize.load(os.path.join(model_path, "3_Normalize"))

		return modules
