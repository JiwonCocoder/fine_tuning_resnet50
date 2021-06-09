import torch.nn as nn

def choose_network(net_from_name, 
                   net,
                   pretrained_from='scratch',
                   pretrained_model_dir='./pretrained_models'):
    # generating models from torchvision.models
    if net_from_name:
        import torchvision.models as models
        model_name_list = sorted(name for name in models.__dict__
                                 if name.islower() and not name.startswith("__")
                                 and callable(models.__dict__[name]))
        if net not in model_name_list:
            assert Exception(f"[!] Networks\' Name is wrong, check net config, \
                               expected: {model_name_list}  \
                               received: {net}")
        else:
            net_model = models.__dict__[net]
            if pretrained_from == 'scratch':
                model = net_model(pretrained=False, num_classes=10)
                model.fc = nn.Linear(model.fc.in_features, 10)

            elif pretrained_from == 'ImageNet_supervised':
                model = net_model(pretrained=True)
                model.fc = nn.Linear(model.fc.in_features, 10)
            elif "SimCLR" in pretrained_from:
                model = models.resnet50(pretrained=False, num_classes=10)
                dataset = pretrained_from.split('_')[0]
                checkpoint_dir = os.path.join(pretrained_model_dir, dataset)
                checkpoint_file = os.path.join(checkpoint_dir,pretrained_from + ".pth.tar")
                checkpoint = torch.load(checkpoint_file)
                state_dict = checkpoint['state_dict']
                for k in list(state_dict.keys()):
                    if k.startswith('backbone.'):
                        if k.startswith('backbone') and not k.startswith('backbone.fc'):
                            # remove prefix
                            state_dict[k[len("backbone."):]] = state_dict[k]
                            if k == 'backbone.conv1.weight':
                                print("here")
                                print(state_dict[k])
                    del state_dict[k]
                log = model.load_state_dict(state_dict, strict=False)
                assert log.missing_keys == ['fc.weight', 'fc.bias']
            return model
