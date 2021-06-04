import torch
import torch.nn.functional as F

from PIL import Image

import os
import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import torchvision
from torchvision import models
from torchvision import transforms as T

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

test_data_transform = T.Compose([
        T.Resize((128,128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
])

dataset_dir = "/home/ubuntu/BYOL/datasets/in_domain_dataset"

test_imgs_dir = os.path.join(dataset_dir, "test")
test_labels = pd.read_csv(os.path.join(dataset_dir, "label/test_label.csv"))

test_set = FineTuneImageDataset(test_labels, test_imgs_dir, transform=test_data_transform)


model_name = "finetuned"

resnet = models.resnet18(pretrained=False)
resnet.fc = torch.nn.Linear(resnet.fc.in_features, 7)

"""
model = resnet
model_location = os.path.join('runs/baselines', f'{args.model_name}.pth')

load_params = torch.load(model_location, map_location=torch.device(torch.device(device)))
model.load_state_dict(load_params['model_state_dict'])
"""

finetuning_encoder = torch.nn.Sequential(*list(resnet.children())[:-1])

"""
            encoder = ResNet18(**config['network'])
            load_params = torch.load(os.path.join('/home/ubuntu/BYOL/runs/pretrained_resnets/checkpoints/model_3.pth'),
                         map_location=torch.device(torch.device(device)))

            if 'online_network_state_dict' in load_params:
                encoder.load_state_dict(load_params['online_network_state_dict'])
"""

output_dim = 7
#output_feature_dim = encoder.projection.net[0].in_features
output_feature_dim = resnet.fc.in_features
model = FineTuningModel(finetuning_encoder, output_feature_dim, output_dim)
model_location = os.path.join('runs/finetuned', f'{model_name}.pth')
load_params = torch.load(model_location, map_location=torch.device(torch.device(device)))
model.load_state_dict(load_params['model_state_dict'])

model.to(device)


