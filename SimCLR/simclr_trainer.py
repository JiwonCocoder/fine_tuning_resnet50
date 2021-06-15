import pytorch_lightning as pl
from pl_bolts.models.self_supervised import SimCLR
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.models.self_supervised.simclr.transforms import (
    SimCLREvalDataTransform, SimCLRTrainDataTransform)
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pl_bolts.transforms.dataset_normalizations import (
    cifar10_normalization,
    imagenet_normalization,
    stl10_normalization,
)
from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, STL10DataModule
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader
def main(args):
    # data
    if args.dataset == 'stl10':
        dm = STL10DataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

        dm.train_dataloader = dm.train_dataloader_mixed
        dm.val_dataloader = dm.val_dataloader_mixed
        args.num_samples = dm.num_unlabeled_samples

        args.maxpool1 = False
        args.first_conv = True
        args.input_height = dm.size()[-1]

        normalization = stl10_normalization()

        args.gaussian_blur = True
        args.jitter_strength = 1.
    elif args.dataset == 'cifar10':
        print(args.data_dir)
        val_split = 5000
        if args.num_nodes * args.gpus * args.batch_size > val_split:
            val_split = args.num_nodes * args.gpus * args.batch_size

        dm = CIFAR10DataModule(
            data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, val_split=val_split
        )

        args.num_samples = dm.num_samples

        args.maxpool1 = False
        args.first_conv = False
        args.input_height = dm.size()[-1]
        args.temperature = 0.5

        normalization = cifar10_normalization()

        args.gaussian_blur = False
        args.jitter_strength = 0.5
        dm.train_transforms = SimCLRTrainDataTransform(32)
        dm.val_transforms = SimCLREvalDataTransform(32)
    elif args.dataset == 'imagenet':
        args.maxpool1 = True
        args.first_conv = True
        normalization = imagenet_normalization()

        args.gaussian_blur = True
        args.jitter_strength = 1.

        args.batch_size = 64
        args.num_nodes = 8
        args.gpus = 8  # per-node
        args.max_epochs = 800

        args.optimizer = 'lars'
        args.learning_rate = 4.8
        args.final_lr = 0.0048
        args.start_lr = 0.3
        args.online_ft = True

        dm = ImagenetDataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

        args.num_samples = dm.num_samples
        args.input_height = dm.size()[-1]
    elif args.dataset == 'MLCC':
        dm = ImageFolder('/data/samsung/Data', transform=SimCLRTrainDataTransform(128, 200)) #(unsup)
        dataset_count=len(dm)
        val_split = 5000
        train_split = dataset_count - val_split
        train_dataset, val_dataset = random_split(dm, [train_split, val_split])
        train_loader = DataLoader(train_dataset, batch_size = args.batch_size)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    else:
        raise NotImplementedError("other datasets have not been implemented till now")



    #checkpoint_callback
    checkpoint_callback = ModelCheckpoint(
        dirpath = os.path.join('/data/simclr_scratch_checkpoints'),
        verbose = True,
        save_last = True,
        save_top_k= 5000,
        monitor = 'val_loss',
        mode = 'min'
    )
    trainer_args = {
        'callbacks': [checkpoint_callback],
        'max_epochs': args.max_epochs,
        'gpus':args.gpus
    }


    # model
    model = SimCLR(num_samples=train_split, batch_size=args.batch_size, dataset=args.dataset, gpus=1)

    # fit
    trainer = pl.Trainer(**trainer_args)
    if args.dataset == 'cifar10':
        trainer.fit(model, datamodule=dm) #cifar
    elif args.dataset == 'MLCC':
        trainer.fit(model, train_loader, val_loader)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')

    # model params
    parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
    # specify flags to store false
    parser.add_argument("--first_conv", action='store_false')
    parser.add_argument("--maxpool1", action='store_false')
    parser.add_argument("--hidden_mlp", default=2048, type=int, help="hidden layer dimension in projection head")
    parser.add_argument("--feat_dim", default=128, type=int, help="feature dimension")
    parser.add_argument("--online_ft", action='store_true')
    parser.add_argument("--fp32", action='store_true')

    # transform params
    parser.add_argument("--gaussian_blur", action="store_true", help="add gaussian blur")
    parser.add_argument("--jitter_strength", type=float, default=1.0, help="jitter strength")
    parser.add_argument("--dataset", type=str, default="MLCC", help="stl10, cifar10, MLCC")
    parser.add_argument("--data_dir", type=str, default="./data", help="path to download data")

    # training params
    parser.add_argument("--fast_dev_run", default=1, type=int)
    parser.add_argument("--num_nodes", default=1, type=int, help="number of nodes for training")
    parser.add_argument("--gpus", default=1, type=int, help="number of gpus to train on")
    parser.add_argument("--num_workers", default=8, type=int, help="num of workers per GPU")
    parser.add_argument("--optimizer", default="adam", type=str, help="choose between adam/lars")
    parser.add_argument('--exclude_bn_bias', action='store_true', help="exclude bn/bias from weight decay")
    parser.add_argument("--max_epochs", default=5000, type=int, help="number of total epochs to run")
    parser.add_argument("--max_steps", default=-1, type=int, help="max steps")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size per gpu")

    parser.add_argument("--temperature", default=0.1, type=float, help="temperature parameter in training loss")
    parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="base learning rate")
    parser.add_argument("--start_lr", default=0, type=float, help="initial warmup learning rate")
    parser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate")


    args = parser.parse_args()
    main(args)