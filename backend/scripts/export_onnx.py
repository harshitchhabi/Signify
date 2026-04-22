import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from landmark_model import LandmarkASLModel
import json

def export_onnx():
    with open("data/label_map.json") as f:
        label_map = json.load(f)
    num_classes = len(label_map)

    model = LandmarkASLModel(num_classes=num_classes)
    model.load_state_dict(torch.load("checkpoints/landmark_model.pt", map_location="cpu"))
    model.eval()

    # The model expects -> (batch, seq_len, 63)
    dummy_input = torch.randn(1, 16, 63)

    onnx_path = "checkpoints/landmark_model.onnx"
    torch.onnx.export(model, dummy_input, onnx_path, 
                      export_params=True, opset_version=11, 
                      do_constant_folding=True, 
                      input_names=['input_sequences'], output_names=['logits'])
    print(f"Successfully exported model to {onnx_path} for Netron!")

if __name__ == "__main__":
    export_onnx()
