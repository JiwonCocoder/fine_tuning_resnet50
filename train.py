from __future__ import division
import os.path as osp
import time
import mmcv
# from utils_from_git import collect_env
from utils_from_git.logger import get_root_logger
# import needed library
import os
import logging
import random
import warnings

import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from models.builder_models import build_model
from datasets.builder_dataset import build_dataset, build_dataloader

from utils import get_logger
from train_utils import TBLog, get_SGD, get_cosine_schedule_with_warmup


def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main(args):
    # log some basic info
    mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, 'train_{}.log'.format(timestamp))
    logger = get_root_logger(log_file=log_file)


    #for checking (later)
    # if args.pretrained is not None:
    #     assert isinstance(args.pretrained, str)
    #     cfg.model.pretrained = args.pretrained
    # SET UP FOR DISTRIBUTED TRAINING

    global best_acc1
    args.gpu = args.gpu

    # random seed has to be set for the syncronization of labeled data sampling in each process.
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    # if args.distributed:
    #     if args.dist_url == "env://" and args.rank == -1:
    #         args.rank = int(os.environ["RANK"])
    #     if args.multiprocessing_distributed:
    #         args.rank = args.rank * args.ngpus_per_node + args.gpu  # compute global rank
    #
    #     # set distributed group:
    #     dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
    #                             world_size=args.world_size, rank=args.rank)

    # SET save_path and logger
    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path) and not args.overwrite:
        raise Exception('already existing model: {}'.format(save_path))
    if args.resume:
        if args.load_path is None:
            raise Exception('Resume of training requires --load_path in the args')
        if os.path.abspath(save_path) == os.path.abspath(args.load_path) and not args.overwrite:
            raise Exception('Saving & Loading pathes are same. \
                               If you want over-write, give --overwrite in the argument.')

    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

        # distributed: true if manually selected or if world_size > 1
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()  # number of gpus of each node

    # divide the batch_size according to the number of nodes
    args.batch_size = int(args.batch_size / args.world_size)
    print("train_batch_size:", args.batch_size)
    if args.multiprocessing_distributed:
        # now, args.world_size means num of total processes in all nodes
        args.world_size = ngpus_per_node * args.world_size

        # args=(,) means the arguments of main_worker
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    '''
       main_worker is conducted on each GPU.
       '''

    global best_acc1
    args.gpu = gpu

    # random seed has to be set for the syncronization of labeled data sampling in each process.
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True

    # SET UP FOR DISTRIBUTED TRAINING
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu  # compute global rank

        # set distributed group:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # SET save_path and logger
    save_path = os.path.join(args.save_dir, args.save_name)
    logger_level = "WARNING"
    tb_log = None
    if args.rank % ngpus_per_node == 0:
        tb_log = TBLog(save_path, 'tensorboard')
        logger_level = "INFO"

    logger = get_logger(args.save_name, save_path, logger_level)
    logger.warning(f"USE GPU: {args.gpu} for training")
    model = build_model(args, tb_log, logger)
    dset_dict = build_dataset(args)
    loader_dict = build_dataloader(args, dset_dict)
    model.set_data_loader(loader_dict)

    # SET Optimizer & LR Scheduler
    ## construct SGD and cosine lr scheduler
    optimizer = get_SGD(model.train_model, 'SGD', args.lr, args.momentum, args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                args.num_train_iter,
                                                num_warmup_steps=args.num_train_iter*0)
    ## set SGD and cosine lr on FixMatch
    model.set_optimizer(optimizer, scheduler)
    # SET Devices for (Distributed) DataParallel
    if not torch.cuda.is_available():
        raise Exception('ONLY GPU TRAINING IS SUPPORTED')
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)

            '''
            batch_size: batch_size per node -> batch_size per gpu
            workers: workers per node -> workers per gpu
            '''
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model.train_model.cuda(args.gpu)
            model.train_model = torch.nn.parallel.DistributedDataParallel(model.train_model,
                                                                          device_ids=[args.gpu])
            model.eval_model.cuda(args.gpu)

        else:
            # if arg.gpu is None, DDP will divide and allocate batch_size
            # to all available GPUs if device_ids are not set.
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.train_model = model.train_model.cuda(args.gpu)
        model.eval_model = model.eval_model.cuda(args.gpu)

    else:
        model.train_model = torch.nn.DataParallel(model.train_model).cuda()
        model.eval_model = torch.nn.DataParallel(model.eval_model).cuda()

    logger.info(f"model_arch: {model}")
    logger.info(f"Arguments: {args}")

    cudnn.benchmark = True

    '''
    resume option
    # If args.resume, load checkpoints from args.load_path
    if args.resume:
        model.load_model(args.load_path)
    '''
    trainer = model.train
    for epoch in range(args.epoch):
        trainer(args, logger=logger)

    if not args.multiprocessing_distributed or \
            (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0):
        model.save_model('latest_model.pth', save_path)

    logging.warning(f"GPU {args.rank} training is FINISHED")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')

    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--work_dir', type=str, default='./log')
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--save_name', type=str, default='fixmatch')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--overwrite', action='store_true')

    '''
    Training Configuration of FixMatch
    '''

    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--num_train_iter', type=int, default=100000,
                        help='total number of training iterations')
    parser.add_argument('--num_eval_iter', type=int, default=10,
                        help='evaluation frequency')
    parser.add_argument('--num_labels', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=4,
                        help='total number of batch size of labeled data')
    parser.add_argument('--uratio', type=int, default=1,
                        help='the ratio of unlabeled data to labeld data in each mini-batch')
    parser.add_argument('--eval_batch_size', type=int, default=2,
                        help='batch size of evaluation data loader (it does not affect the accuracy)')

    parser.add_argument('--hard_label', type=bool, default=True)
    parser.add_argument('--T', type=float, default=0.5)
    parser.add_argument('--p_cutoff', type=float, default=0.95)
    parser.add_argument('--ema_m', type=float, default=0.999, help='ema momentum for eval_model')
    parser.add_argument('--ulb_loss_ratio', type=float, default=1.0)

    '''
    Optimizer configurations
    '''
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--amp', action='store_true', help='use mixed precision training or not')

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--learning_type', type=str, default=None, help='semi|sup')
    parser.add_argument('--baseline', type=str, default=None, help='Fixmatch|KD_distitll|supervised')
    parser.add_argument('--net', type=str, default='WideResNet')
    parser.add_argument('--net_from_name', type=bool, default=False)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)

    '''
    pretrained_model Configurations
    '''
    parser.add_argument('--pretrained_from', type=str, default='scratch',
                        help='scratch | ImageNet_supervised | ImageNet_SimCLR | CIFAR_SimCLR | MLCC_SimCLR')
    parser.add_argument('--pretrained_model_dir', type=str, default='pretrained_model',
                        help='pretrained_model_dir')
    parser.add_argument('--fine_tuning', type=str, default=None,
                        help='supervised | semi_supervised')
    '''
    for KD_distillation
    '''
    parser.add_argument('--teacher_net', type=str, default=None,
                        help='resnet50'
                        )
    parser.add_argument('--student_net', type=str, default=None,
                        help='resnet50')
    parser.add_argument('--alpha', default=None, type=float,
                        help="loss_kd")
    parser.add_argument('--temperature', default=None, type=float,
                        help="loss_kd")

    '''
    Data Configurations
    '''

    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10 | STL10 | SVHN | MLCC')
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=1)

    '''
    multi-GPUs & Distrbitued Training--eval_batch_size 10
    '''

    ## args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='**node rank** for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:10001', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    parser.add_argument('--mixup', default=0, type=int, help='wanna mix ? 1 == mixup | 0 == non mixup')

    args = parser.parse_args()
    main(args)