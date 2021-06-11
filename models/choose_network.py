import torch.nn as nn
import pdb
import os
import torch
def choose_network(args, net_from_name,
                   net,
                   pretrained_from='scratch',
                   pretrained_model_dir='./pretrained_models',
                   ):
    # generating models from torchvision.models
    if net_from_name:
        import torchvision.models as models
        model_name_list = sorted(name for name in models.__dict__
                                 if name.islower() and not name.startswith("__")
                                 and callable(models.__dict__[name]))
        if net not in model_name_list: #토치비전에 모델 없으면 에러
            assert Exception(f"[!] Networks\' Name is wrong, check net config, \
                               expected: {model_name_list}  \
                               received: {net}")
        else:
            # model = models.__dict__['wide_resnet50_2'](pretrained=True)
            # model.fc = nn.Linear(model.fc.in_features, 10)
            # return model

            net_model = models.__dict__[net]
            print(net_model.__name__+" is used")
            if pretrained_from == 'scratch':
                model = net_model(pretrained=False, num_classes=args.num_classes)
                model.fc = nn.Linear(model.fc.in_features, args.num_classes)
            elif pretrained_from == 'ImageNet_supervised':
                model = net_model(pretrained=True)
                model.fc = nn.Linear(model.fc.in_features, args.num_classes)
            elif "SimCLR" in pretrained_from:# CAUTION : 와이드레즈넷일때 따로 추가해줘야함.
                model = models.resnet50(pretrained=False, num_classes=args.num_classes)
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
    else: # if net_from_name == false 인 경우
        # 여기도 if 문으로 프리트레인 조건 넣어줘야함. 그러나 net from name 을 항상 true 로 두면 그럴 필요 없음
        model = models.__dict__['wide_resnet50_2'](pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
        return model

