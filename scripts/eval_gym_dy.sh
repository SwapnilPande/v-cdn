CUDA_VISIBLE_DEVICES=2 \
	python eval_dy.py \
	--env half-cheetah \
	--stage dy \
	--baseline 0 \
	--gauss_std 5e-2 \
	--lam_kp 10 \
	--en_model cnn \
	--dy_model gnn \
	--preload_kp 1 \
	--nf_hidden_kp 16 \
	--nf_hidden_dy 16 \
	--n_kp 7 \
	--inv_std 10 \
	--min_res 46 \
	--n_identify 100 \
	--n_his 1 \
	--n_roll 5 \
	--node_attr_dim 0 \
	--edge_attr_dim 1 \
	--edge_type_num 2 \
	--edge_st_idx 1 \
	--edge_share 1 \
	--batch_size 1 \
	--lr 1e-4 \
	--gen_data 0 \
	--num_workers 10 \
	--kp_epoch 2 \
	--kp_iter 10000 \
	--dy_epoch -1 \
	--dy_iter -1 \
	--log_per_iter 50 \
	# --eval 1
