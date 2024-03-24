export CUDA_VISIBLE_DEVICES='1,2'
python train_abr.py --exp_name=xpoolMR9k_compress_trail2 \
                    --videos_dir=preprocess/all_compress/ \
                    --batch_size=16 \
                    --noclip_lr=3e-5 \
                    --transformer_dropout=0.3 \
                    --huggingface --dataset_name=MSRVTTABR \
                    --msrvtt_train_file=9k \
                    --loss ABR \
                    --arch clip_abr_transformer \
                    # --framecap_loss_weight 0.00 \
                    # --clip_lr 0.0 \
                    # --noclip_lr 0.0 \
                    # --weight_decay 0.0000