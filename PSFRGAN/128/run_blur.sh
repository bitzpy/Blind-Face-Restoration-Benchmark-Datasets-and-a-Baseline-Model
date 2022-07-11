python train.py --gpus 1 --model enhance --name blur \
    --g_lr 0.0001 --d_lr 0.0004 --beta1 0.5 \
    --gan_mode 'hinge' --lambda_pix 10 --lambda_fm 10 --lambda_ss 1000 \
    --Dinput_nc 22 --D_num 3 --n_layers_D 2 \
    --batch_size 8 --dataset ffhq   \
    #--continue_train