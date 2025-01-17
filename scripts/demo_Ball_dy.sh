CUDA_VISIBLE_DEVICES=0 \
	python eval_dy.py \
	--env Ball \
	--stage dy \
	--baseline 0 \
	--gauss_std 5e-2 \
	--lam_kp 1e1 \
	--en_model cnn \
	--dy_model gnn \
	--nf_hidden_kp 16 \
	--nf_hidden_dy 16 \
	--n_kp 5 \
	--inv_std 10 \
	--min_res 46 \
	--n_identify 100 \
	--n_his 10 \
	--n_roll 5 \
	--node_attr_dim 0 \
	--edge_attr_dim 1 \
	--edge_type_num 3 \
	--edge_st_idx 1 \
	--edge_share 1 \
	--eval_set demo \
	--eval_st_idx 100 \
	--eval_ed_idx 150 \
	--identify_st_idx 0 \
	--identify_ed_idx 100 \
	--eval_kp_epoch 2 \
	--eval_kp_iter 10000 \
	--eval_dy_epoch 2 \
	--eval_dy_iter 60000 \
	--store_demo 1 \
	--vis_edge 1 \
