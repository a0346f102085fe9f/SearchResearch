from transformers import AutoTokenizer
from multiprocessing import Pool
import torch
import sys
import os

# File-aware parallelism manually implemented here
os.environ["TOKENIZERS_PARALLELISM"] = "false"

tokenizer = AutoTokenizer.from_pretrained("hkunlp/instructor-base")

# Bag of tokens tool
# Takes a folder of .txt files
if len(sys.argv) < 2:
	print("Usage: python3 bincount.py Folder/")
	exit()

path = sys.argv[1]

def bincount_file(filename):
	file = open(path + filename, encoding='utf-8', errors='replace')
	text = file.read()
	file.close()

	tokens, _ = tokenizer(text, return_tensors='pt', add_special_tokens=False).values()
	vocab_size = tokenizer.vocab_size

	# Workaround for empty documents
	if tokens.dtype == torch.float:
		return bytes(vocab_size*4)

	# Pytorch screams at you a little for doing this
	# Still, this is the most performant way of getting raw tensor data out of it
	data = bytes(vocab_size*4)
	bins = torch.frombuffer(data, dtype=torch.int32)

	bins += torch.bincount(tokens[0], minlength=vocab_size)

	return data


# Progress meter for pool.imap()
class progress():
	def __init__(self, total):
		self.total = total
		self.i = 0

	def update(self, iterator):
		mapped = []

		for x in iterator:
			mapped.append(x)
			self.i += 1
			print(f"{self.i}/{self.total}")

		return mapped

if __name__ == '__main__':
	dir = os.listdir(path)

	idx = open("src_titles.txt", "w", encoding='utf-8', errors='replace')
	idx.write("\n".join(dir))
	idx.close()

	# Bincount datatape contains int32 values, each being a counter for some token hitcount in a document
	datatape_src = open("datatape_src.bin", "wb")

	status = progress(len(dir))

	with Pool(8) as pool:
		bins = status.update(pool.imap(bincount_file, dir))

	for data in bins:
		datatape_src.write(data)

	datatape_src.close()
