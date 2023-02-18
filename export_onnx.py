from models import instructor as net
from onnxruntime.quantization import quantize_dynamic, QuantType
import torch

input = {
	"input_ids": torch.ones([1, 512], dtype=torch.int32),
	"attention_mask": torch.ones([1, 512], dtype=torch.int32)
}

dynamic_axes = {
	"input_ids": [1],
	"attention_mask": [1]
}

print("Exporting...")
torch.onnx.export(net.model, {"input": input}, "instructor.onnx", input_names = ["input_ids", "attention_mask"], output_names = ["result"], dynamic_axes=dynamic_axes)

print("Quantizing...")
quantize_dynamic("instructor.onnx", "instructor-uint8.onnx", weight_type=QuantType.QUInt8, extra_options={"MatMulConstBOnly":False})
