python training.py --prefix_name resnet_51_extractor_w_tut_256_256_180_180_pi_and_180_180_32_vf --pi 256 256 180 180 --vf 180 180 32 \
--mid_channels 20 --num_first_cnn_layer 16 --n_steps 32 --train_steps 2 --batch_size 128 --num_workers 8  --lr 0.00003  \
--device cpu --epochs 3 \
--checkpoint /Users/hung/Documents/coding/internVNG/TinkleMatch3/_saved_model/resnet_51_extractor_w_tut_256_256_180_180_pi_and_180_180_32_vf_16layers_20channels_3e-05_32_not_share_20240802.pt\