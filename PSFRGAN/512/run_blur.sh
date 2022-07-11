python train.py --gpus 2 --model enhance --name blur \
    --g_lr 0.001 --d_lr 0.004 --beta1 0.5 \
    --gan_mode 'hinge' --lambda_pix 10 --lambda_fm 10 --lambda_ss 1000 \
    --Dinput_nc 22 --D_num 3 --n_layers_D 4 \
    --batch_size 2 --dataset ffhq  \
     #--continue_train