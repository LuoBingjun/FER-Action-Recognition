import torch
import torchvision.models as models
from model.resnext import *


def get_action_classifier(arch='resnext50_32x4d', base_width=4, cardinality=32, num_classes=4,
              checkpoint_path='./ckpt/resnext50-32x4d/res3/model_best.pth.tar'):
    model = resnext50(
        baseWidth=base_width,
        cardinality=cardinality,
        num_classes=num_classes
    )
    if arch.startswith('alexnet') or arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    model.eval()

    return model
