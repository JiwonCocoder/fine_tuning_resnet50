#Cifar10 / supervised /ImageNet_supervised 
python train.py --epoch 10000 --batch_size 16 --eval_batch_size 10 --num_eval_iter 100\
                --lr 0.002 \
                --learning_type sup --baseline supervised --net resnet50 --net_from_name True \
                --pretrained_from ImageNet_supervised --mixup 0 \
                --data_dir ./data --dataset cifar10 --num_classes 10 \
                --world-size 1 --rank 0 --gpu 1

#MLCC / supervised /ImageNet_supervised 
python train.py --epoch 10000 --batch_size 16 --eval_batch_size 10 --num_eval_iter 100\
                --lr 0.002 \
                --learning_type sup --baseline supervised --net resnet50 --net_from_name True \
                --pretrained_from ImageNet_supervised --mixup 0 \
                --data_dir /root/dataset2/Samsung_labeled_only --dataset MLCC --num_classes 10 \
                --world-size 1 --rank 0 --gpu 1 




# python train.py --num_train_iter 20000 --num_labels 300 --batch_size 16 --eval_batch_size 10 \
#                  --lr 0.002 \
#                  --learning_type semi --baseline Fixmatch --net resnet50 --net_from_name True \
#                  --pretrained_from ImageNet_SimCLR --mixup 0 \
#                  --data_dir /root/dataset2/Samsung_fixmatch --dataset MLCC --train_sampler RandomSampler --num_classes 10 \
#                  --world-size 1 --rank 0 --gpu 1


# python train.py --num_train_iter 20000 --num_labels 4000 --batch_size 16 --eval_batch_size 10 \
#                  --lr 0.002 \
#                  --learning_type semi --baseline Fixmatch --net resnet50 --net_from_name True \
#                  --pretrained_from ImageNet_supervised --mixup 0 \
#                  --data_dir ./data --dataset cifar10 --num_classes 10 \
#                  --world-size 1 --rank 0 --gpu 1