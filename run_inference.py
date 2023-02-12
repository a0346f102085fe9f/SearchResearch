from array import array
import sys
import os

#from models import cocodr as model
from models import instructor as net

# Inference tool
# Takes a folder of .txt files
if len(sys.argv) < 2:
	print("Usage: python3 run_inference.py Folder/")
	exit()

datatape_v = open("datatape_v.bin", "wb")

path = sys.argv[1]
dir = os.listdir(path)
i = 0

idx = open("idx.json", "w", encoding='utf-8', errors='replace')
idx.write("{")

for filename in dir:
	file = open(path + filename, encoding='utf-8', errors='replace')
	text = file.read()
	file.close()

	data = net.run(text)

	v_array = array("f", data)
	v_array.tofile(datatape_v)

	json = "\"" + filename + "\":{\"dimensions\":" + str(len(data)) + "}"
	idx.write(json)

	# Put a comma unless this is the last file
	if (len(dir) - i) > 1:
		idx.write(",")

	i = i + 1
	print(str(i) + "/" + str(len(dir)) + " " + filename)

idx.write("}")
idx.close()

datatape_v.close()
