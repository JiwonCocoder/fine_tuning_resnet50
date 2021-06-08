from .ssl_dataset import SSL_Dataset
from .data_utils import get_data_loader

def build_dataset(args):
    if args.learning_type:
        #Construct Dataset
        train_dset = SSL_Dataset(name=args.dataset, learning_type=args.learning_type, train=True,
                                 num_classes=args.num_classes, data_dir=args.data_dir)
        #Return lb_dataset, ulb_dataset
        lb_dset, ulb_dset = train_dset.get_ssl_dset(args.num_labels)

        _eval_dset = SSL_Dataset(name=args.dataset, learning_type=args.learning_type, train=False,
                                 num_classes=args.num_classes, data_dir=args.data_dir)
        eval_dset = _eval_dset.get_dset()
        loader_dict = {}
        dset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset, 'eval': eval_dset}

    # else:
    #     dset_dict = {'train:', 'eval:'}

    return dset_dict

def build_dataloader(args, dset_dict):
    loader_dict = {}
    if args.learning_type == 'semi':
        loader_dict['train_lb'] = get_data_loader(dset_dict['train_lb'],
                                                  args.batch_size,
                                                  data_sampler=args.train_sampler,
                                                  num_iters=args.num_train_iter,
                                                  num_workers=args.num_workers,)
                                                  # distributed=args.distributed)

        loader_dict['train_ulb'] = get_data_loader(dset_dict['train_ulb'],
                                                   args.batch_size * args.uratio,
                                                   data_sampler=args.train_sampler,
                                                   num_iters=args.num_train_iter,
                                                   num_workers=4 * args.num_workers,)
                                                   # distributed=args.distributed)

        loader_dict['eval'] = get_data_loader(dset_dict['eval'],
                                              args.eval_batch_size,
                                              num_workers=args.num_workers)


    elif args.learning_type =='sup':
        loader_dict['train_lb'] = get_data_loader(dset_dict['train'],
                                                  args.batch_size,
                                                  num_workers=args.num_workers
                                                  )
        loader_dict['eval'] = get_data_loader(dset_dict['eval'],
                                              args.eval_batch_size,
                                              num_workers=args.num_workers)
    return loader_dict