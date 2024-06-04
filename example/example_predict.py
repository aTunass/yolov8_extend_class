import copy
import torch
from ultralytics import YOLO
from PIL import Image
import os
def compare_dicts(state_dict1, state_dict2):
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())
    if keys1 != keys2:
        print("Models have different parameter names.")
        return False
    for key in keys1:
        if not torch.equal(state_dict1[key], state_dict2[key]):
            print(f"Weights for parameter '{key}' are different.")
            if "bn" in key and "22" not in key:
              state_dict1[key] = state_dict2[key]
if __name__ == "__main__":
    """check"""
    model = YOLO('first_weight.pt') 
    model2 = YOLO('second_weight.pt') 
    compare_dicts(model2.state_dict(), model.state_dict())
    
    """save dict"""
    new_state_dict = dict()
    model = YOLO('second_weight.pt')
    for k, v in model.state_dict().items():
        if k.startswith("model.model.22"):
            new_state_dict[k.replace("model.22", "model.23")] = v
    torch.save(new_state_dict, "second_weight_head.pth")
    
    """load model"""
    model = YOLO('example_config.yaml', task="detect").load('first_weight.pt')
    label_dict = {}
    with open('example_class_name.txt', 'r') as file:
        for line_number, line in enumerate(file, 0):  # Start line numbering from 1
            parts = line.strip().split(':')
            label_name = parts[1].strip()
            label_dict[line_number] = label_name
    model.model.names = label_dict
    state_dict = torch.load("second_weight_head.pth")
    model.load_state_dict(state_dict, strict=False)
    """Predict"""
    source = 'https://genk.mediacdn.vn/139269124445442048/2022/6/19/kiskreations15625694511150307856119168285573707314030520n-1655613685894711491522.jpg'
    results = model.predict([source], conf=0.1, iou=0.7) 
    for i, r in enumerate(results):
        print("boxes",r.boxes.xywh) 
        print("conf",r.boxes.conf) 
        im_bgr = r.plot()  # BGR-order numpy array
        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
        r.save(filename=f'results{i}.jpg')