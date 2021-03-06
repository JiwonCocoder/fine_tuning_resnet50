# rm -r ./saved_models/scratch_300_002/
#WideResNet
# CUDA_VISIBLE_DEVICES=1 python train_copy2.py --world-size 1 --rank 0 --num_labels 300 --save_name scratch_300_002 --dataset cifar10 --num_classes 10 --lr 0.002 --multiprocessing-distributed
#ResNet
# CUDA_VISIBLE_DEVICES=1 python train_copy2.py --world-size 1 --rank 0 --num_labels 300 --save_name scratch_300_002 --dataset cifar10 --num_classes 10 --lr 0.002 --multiprocessing-distributed --net resnet50 --net_from_name True
#python train.py --world-size 1 --rank 0 --num_labels 300 --save_name scratch_300_0002 --data_dir ./data  --dataset cifar10 --num_classes 10 --lr 0.0002 --net resnet50 --net_from_name True --eval_batch_size 10 --gpu 0 --pretrained_from scratch --pretrained_model_dir pretrained_model --learning_type semi --baseline Fixmatch

#for KD
rm -r ./saved_models

python train.py --num_train_iter 20000 -num_eval_iter 100 --num_labels 300 --batch_size 2 --eval_batch_size 10 \
                 --lr 0.002 \
                 --learning_type semi --baseline KD_distill --student_net resnet50 --net_from_name True --temperature 0.01 --alpha 0.35\
                 --teacher_net resnet50 --pretrained_from ImageNet_supervised --mixup 0 \
                 --data_dir ./data --dataset cifar10 --train_sampler RandomSampler --num_classes 10 \
                 --world-size 1 --rank 0 --gpu 0


#MLCC / KD_distill /supervised_imageNet_finetuning
#python train.py --num_train_iter 20000 -num_eval_iter 100 --num_labels 300 --batch_size 2 --eval_batch_size 10 \
#                 --lr 0.002 \
#                 --learning_type semi --baseline KD_distill --student_net resnet50 --net_from_name True --temperature 0.01 --alpha 0.35\
#                 --teacher_net resnet50 --pretrained_from ImageNet_supervised --mixup 0 \
#                 --data_dir ./data --dataset cifar10 --train_sampler RandomSampler --num_classes 10 \
#                 --world-size 1 --rank 0 --gpu 0


#MLCC / Fixmatch /supervised_imageNet_finetuning
#python train.py --num_train_iter 20000 --num_labels 300 --batch_size 64 --eval_batch_size 10 \
#                 --lr 0.002 \
#                 --learning_type semi --baseline Fixmatch --net resnet50 --net_from_name True \
#                 --pretrained_from ImageNet_supervised --mixup 0 \
#                 --data_dir ./data --dataset MLCC --train_sampler RandomSampler --num_classes 10 \
#                 --world-size 1 --rank 0 --gpu 0


##CIFAR10 / Fixmatch /supervised_imageNet_finetuning
#python train.py --num_train_iter 20000 --num_labels 4000 --batch_size 64 --eval_batch_size 10 \
#                 --lr 0.002 \
#                 --learning_type semi --baseline Fixmatch --net resnet50 --net_from_name True \
#                 --pretrained_from ImageNet_supervised --mixup 0 \
#                 --data_dir ./data --dataset cifar10 --train_sampler RandomSampler --num_classes 10 \
#                 --world-size 1 --rank 0 --gpu 0

##CIFAR10 / Fixmatch / simclr_imageNet finetuning
#python train.py --num_train_iter 20000 --num_labels 4000 --batch_size 64 --eval_batch_size 10 \
#                 --lr 0.002 \
#                 --learning_type semi --baseline Fixmatch --net resnet50 --net_from_name True \
#                 --pretrained_from ImageNet_SimCLR --mixup 0 \
#                 --data_dir ./data --dataset cifar10 --train_sampler RandomSampler --num_classes 10 \
#                 --world-size 1 --rank 0 --gpu 0



#python train.py  --num_train_iter 1000 --batch_size 4 --eval_batch_size 10 \
#                 --lr 0.002 \
#                 --learning_type sup --baseline supervised --net wide_resnet50_2 --net_from_name True \
#                 --pretrained_from ImageNet_supervised --mixup 0 \
#                 --data_dir ./data --dataset cifar100 --train_sampler RandomSampler --num_classes 100 \
#                 --world-size 1 --rank 0 --gpu 0
#
# python train.py  --num_train_iter 1000 --num_labels 4000 --batch_size 128 --eval_batch_size 100 \
#                 --lr 0.002 \
#                 --learning_type semi --baseline Fixmatch --net wide_resnet50_2 --net_from_name True \
#                 --pretrained_from ImageNet_supervised --mixup 0 \
#                 --data_dir ./data --dataset cifar10 --train_sampler RandomSampler --num_classes 10 \
#                 --world-size 1 --rank 0 --gpu 0
#
# python train.py  --num_train_iter 1000 --batch_size 128 --eval_batch_size 100 \
#                 --lr 0.002 \
#                 --learning_type sup --baseline supervised --net wide_resnet50_2 --net_from_name True \
#                 --pretrained_from ImageNet_supervised --mixup 0 \
#                 --data_dir ./data --dataset cifar10 --train_sampler RandomSampler --num_classes 10 \
#                 --world-size 1 --rank 0 --gpu 0
#



#--num_train_iter 1000 --num_labels 4000 --batch_size 128 --eval_batch_size 100 --lr 0.002 --learning_type semi --baseline Fixmatch --net wide_resnet50_2 --net_from_name True --pretrained_from ImageNet_supervised --mixup 0 --data_dir ./data --dataset cifar10 --train_sampler RandomSampler --num_classes 10 --world-size 1 --rank 0 --gpu 0

#python train.py --world-size 1 --rank 0 --num_labels 300 --save_name scratch_300_0002 --data_dir /data/samsung  --dataset MLCC --num_classes 10 --lr 0.0002 --net resnet50 --net_from_name True --eval_batch_size 10 --gpu 0\
# --pretrained_from scratch --pretrained_model_dir pretrained_model --learning_type semi --baseline Fixmatch
#python train.py --world-size 1 --rank 0 --num_labels 300 --save_name scratch_300_0002 --data_dir /data/samsung  --dataset MLCC --num_classes 10 --lr 0.0002 --net resnet50 --net_from_name True --eval_batch_size 10 --gpu 0\
# --pretrained_from scratch --pretrained_model_dir pretrained_model --learning_type semi --baseline Fixmatch

# python train_copy2.py --world-size 1 --rank 0 --num_labels 300 --save_name test --dataset cifar10 --num_classes 10 --lr 0.0004 --net resnet50 --net_from_name True --eval_batch_size 10
# python train_copy2.py --world-size 1 --rank 0 --num_labels 300 --save_name scratch_300_0001 --dataset cifar10 --num_classes 10 --lr 0.0001 --net resnet50 --net_from_name True --eval_batch_size 10 --gpu 1
# python train_copy2.py --world-size 1 --rank 0 --num_labels 300 --save_name scratch_300_002 --dataset cifar10 --num_classes 10 --lr 0.002 --net resnet50 --net_from_name True --eval_batch_size 10 --gpu 1
# python train_copy2.py --world-size 1 --rank 0 --num_labels 300 --save_name scratch_300_001 --dataset cifar10 --num_classes 10 --lr 0.001 --net resnet50 --net_from_name True --eval_batch_size 10 --gpu 1
# python train_copy2.py --world-size 1 --rank 0 --num_labels 300 --save_name scratch_300_01 --dataset cifar10 --num_classes 10 --lr 0.01 --net resnet50 --net_from_name True --eval_batch_size 10 --gpu 1
# python train_copy2.py --world-size 1 --rank 0 --num_labels 300 --save_name scratch_300_02 --dataset cifar10 --num_classes 10 --lr 0.02 --net resnet50 --net_from_name True --eval_batch_size 10 --gpu 1
# python train_copy2.py --world-size 1 --rank 0 --num_labels 300 --save_name scratch_300_05 --dataset cifar10 --num_classes 10 --lr 0.05 --net resnet50 --net_from_name True --eval_batch_size 10 --gpu 1

# python train_copy2.py --world-size 1 --rank 0 --num_labels 300 --save_name test_shouldBeDeled2 --data_dir /data/samsung  --dataset MLCC --num_classes 10 --lr 0.0002 --net resnet50 --net_from_name True --eval_batch_size 10 --gpu 0 --pretrained_from MLCC_SimCLR --pretrained_model_dir pretrained_model

# CUDA_VISIBLE_DEVICES=1 python train_copy2.py --world-size 1 --rank 0 --num_labels 300 --save_name scratch_300_002 --dataset cifar10 --num_classes 10 --lr 0.002 --multiprocessing-distributed --net resnet50 --net_from_name True --eval_batch_size 10
# CUDA_VISIBLE_DEVICES=1 python train_copy2.py --world-size 1 --rank 0 --num_labels 300 --save_name scratch_300_001 --dataset cifar10 --num_classes 10 --lr 0.001 --multiprocessing-distributed --net resnet50 --net_from_name True
# CUDA_VISIBLE_DEVICES=1 python train_copy2.py --world-size 1 --rank 0 --num_labels 300 --save_name scratch_300_01 --dataset cifar10 --num_classes 10 --lr 0.01 --multiprocessing-distributed --net resnet50 --net_from_name True
# CUDA_VISIBLE_DEVICES=1 python train_copy2.py --world-size 1 --rank 0 --num_labels 300 --save_name scratch_300_02 --dataset cifar10 --num_classes 10 --lr 0.02 --multiprocessing-distributed --net resnet50 --net_from_name True
# CUDA_VISIBLE_DEVICES=1 python train_copy2.py --world-size 1 --rank 0 --num_labels 300 --save_name scratch_300_05 --dataset cifar10 --num_classes 10 --lr 0.05 --multiprocessing-distributed --net resnet50 --net_from_name True

# python train_copy2.py --world-size 1 --rank 0 --num_labels 300 --save_name testtest --dataset cifar10 --num_classes 10 --lr 0.05 --net resnet50 --net_from_name True --eval_batch_size 10 --fine_tuning
#python ./sup_set/train.py --decay 0.005 --batch_size 64 --lr 0.002 --seed 20170922 --limit_data False --mixup False --pretrain scratch --dataset MLCC --name test --epoch 200
