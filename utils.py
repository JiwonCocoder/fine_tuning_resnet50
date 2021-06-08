import os
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import logging
import torch
def define_model(net_model, net_from_name, pretrained_from, pretrained_model_dir):
    if net_from_name:
        import torchvision.models as models
        model_name_list = sorted(name for name in models.__dict__
                                if name.islower() and not name.startswith("__")
                                and callable(models.__dict__[name]))

        if net_model not in model_name_list:
            assert Exception(f"[!] Networks\' Name is wrong, check net config, \
                               expected: {model_name_list}  \
                               received: {net_model}")
        else:
            net_model = models.__dict__[net_model]
            if pretrained_from == 'scratch':
                model = net_model(pretrained = False, num_classes=10)
            elif pretrained_from == 'ImageNet_supervised':
                model = net_model(pretrained = True)
                model.fc = nn.Linear(model.fc.in_features, 10)
            elif "SimCLR" in pretrained_from :
                model = models.resnet50(pretrained=False, num_classes=10)
                dataset = pretrained_from.split('_')[0]
                checkpoint_dir = os.path.join(pretrained_model_dir, dataset)
                checkpoint_file = os.path.join(checkpoint_dir, pretrained_from+".pth.tar")
                checkpoint = torch.load(checkpoint_file)
                state_dict = checkpoint['state_dict']
                for k in list(state_dict.keys()):
                    if k.startswith('backbone.'):
                        if k.startswith('backbone') and not k.startswith('backbone.fc'):
                            # remove prefix
                            state_dict[k[len("backbone."):]] = state_dict[k]
                            if k=='backbone.conv1.weight':
                                print("here")
                                print(state_dict[k])
                    del state_dict[k]
                log = model.load_state_dict(state_dict, strict=False)
                assert log.missing_keys == ['fc.weight', 'fc.bias']
        return model

def setattr_cls_from_kwargs(cls, kwargs):
    #if default values are in the cls,
    #overlap the value by kwargs
    for key in kwargs.keys():
        if hasattr(cls, key):
            print(f"{key} in {cls} is overlapped by kwargs: {getattr(cls,key)} -> {kwargs[key]}")
        setattr(cls, key, kwargs[key])

        
def test_setattr_cls_from_kwargs():
    class _test_cls:
        def __init__(self):
            self.a = 1
            self.b = 'hello'
    test_cls = _test_cls()
    config = {'a': 3, 'b': 'change_hello', 'c':5}
    setattr_cls_from_kwargs(test_cls, config)
    for key in config.keys():
        print(f"{key}:\t {getattr(test_cls, key)}")
        
        
def net_builder(net_name, from_name: bool, net_conf=None):
    """
    return **class** of backbone network (not instance).
    Args
        net_name: 'WideResNet' or network names in torchvision.models
        from_name: If True, net_buidler takes models in torch.vision models. Then, net_conf is ignored.
        net_conf: When from_name is False, net_conf is the configuration of backbone network (now, only WRN is supported).
    """
    if from_name:
        import torchvision.models as models
        model_name_list = sorted(name for name in models.__dict__
                                if name.islower() and not name.startswith("__")
                                and callable(models.__dict__[name]))

        if net_name not in model_name_list:
            assert Exception(f"[!] Networks\' Name is wrong, check net config, \
                               expected: {model_name_list}  \
                               received: {net_name}")
        else:
            return models.__dict__[net_name]
        
    else:
        if net_name == 'WideResNet':
            import models.nets.wrn as net
            builder = getattr(net, 'build_WideResNet')()
        else:
            assert Exception("Not Implemented Error")
            
        setattr_cls_from_kwargs(builder, net_conf)
        return builder.build

    
def test_net_builder(net_name, from_name, net_conf=None):
    builder = net_builder(net_name, from_name, net_conf)
    print(f"net_name: {net_name}, from_name: {from_name}, net_conf: {net_conf}")
    print(builder)

    
def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    log_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)
    
    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)
    
    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
