import os
from pl_bolts.models.self_supervised import SimCLR

from pl_bolts.models.self_supervised import SimCLR
weight_path = '/home/ubuntu/fine_tuning_baselines/fine_tuning_resnet50/pretrained_model/resnet50/cifar10/simclr'

weight_name = 'CIFAR10_SimCLR.ckpt'

file_path = os.path.join(weight_path, weight_name)
simclr = SimCLR.load_from_checkpoint(file_path, strict=False)
simclr_resnet50 = simclr.encoder

for name, param in simclr.encoder.named_parameters():
    if param.requires_grad:
        print(name)