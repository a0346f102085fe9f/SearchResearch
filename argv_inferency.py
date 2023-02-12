#from models import cocodr as net
from models import instructor as net
import sys

# Basic model/tokenizer debugging tool
# First argument is the query, the rest are documents
if len(sys.argv) < 2:
	print("Usage: python3 argv_inference.py \"The capital of France\" \"Paris\" \"Berlin\" \"London\"")
	exit()

src = sys.argv[1:]

inspect_ts = [net.inspect(net.tokenize(x)) for x in src]
embeddings = [net.run(x) for x in src]

a = embeddings.pop(0)

scores = [net.similarity(a, b) for b in embeddings]
scores = list(zip(scores, src[1:]))
scores.sort()

for score, text in scores:
	print(score, text)

