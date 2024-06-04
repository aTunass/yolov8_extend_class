from ultralytics import YOLO
import os
import torch
import copy
def put_in_eval_mode(trainer, n_layers=22):
  for i, (name, module) in enumerate(trainer.model.named_modules()):
    if name.endswith("bn") and int(name.split('.')[1]) < n_layers:
      module.eval() 
      module.track_running_stats = False
"""finetune"""
model = YOLO('initial_weight.pt')  # load a pretrained model (recommended for training)
# for param in model.model.model[:10].parameters():
#     param.requires_grad = False
# Train the model with 2 GPUs
old_dict = copy.deepcopy(model.state_dict())
model.add_callback("on_train_epoch_start", put_in_eval_mode)
model.add_callback("on_pretrain_routine_start", put_in_eval_mode)
results = model.train(data='example_config.yaml', epochs=20, imgsz=640, freeze=22,
                       device=3, batch=40, plots=True, name= "train_extend")
