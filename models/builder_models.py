import torch.nn as nn
import os
import torch
from .fixmatch import FixMatch
from .supervisedModel import SupervisedModel

def build_model(args, tb_log, logger):
    if args.baseline == 'Fixmatch':
        model = FixMatch(args,
                     args.num_classes,
                     args.ema_m,
                     args.T,
                     args.p_cutoff,
                     args.ulb_loss_ratio,
                     args.hard_label,
                     num_eval_iter=args.num_eval_iter,
                     tb_log=tb_log,
                     logger=logger)
    elif args.baseline == 'supervised':
        model = SupervisedModel(args,
                     args.num_classes,
                     args.ema_m,
                     args.T,
                     args.p_cutoff,
                     args.ulb_loss_ratio,
                     args.hard_label,
                     num_eval_iter=args.num_eval_iter,
                     tb_log=tb_log,
                     logger=logger)
    
    return model