from .resnet import ResNet
from .alex_net import Alexnet
from .light_net import Light_net


dict_models = {
    'resnet': ResNet,
    'alexnet': Alexnet,
    'lightnet': Light_net
}


# get the model from the dict
def get_backbone_model(flags):
    backbone_name = flags.backbone

    if backbone_name in dict_models.keys():
        model = dict_models[backbone_name]
    else:
        raise ValueError('Unable to find the model {}. The program will exit now'.format(backbone_name))

    return model