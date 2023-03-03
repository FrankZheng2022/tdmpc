### Experiment 2,7
CUDA_VISIBLE_DEVICES=0 python src/train.py task=walker-walk amlt=0 seed=5 latent_action_dim=6 act_reconstruct_coef=1 inv_consistency_coef=1 exp_name=walker_walk_latent6_alpha1_inv1_1 &
CUDA_VISIBLE_DEVICES=1 python src/train.py task=walker-walk amlt=0 seed=6 latent_action_dim=6 act_reconstruct_coef=1 inv_consistency_coef=1 exp_name=walker_walk_latent6_alpha1_inv1_2 &
CUDA_VISIBLE_DEVICES=2 python src/train.py task=walker-walk amlt=0 seed=7 latent_action_dim=6 act_reconstruct_coef=1 inv_consistency_coef=1 exp_name=walker_walk_latent6_alpha1_inv1_3 &
CUDA_VISIBLE_DEVICES=3 python src/train.py task=walker-walk amlt=0 seed=8 latent_action_dim=6 act_reconstruct_coef=1 inv_consistency_coef=1 exp_name=walker_walk_latent6_alpha1_inv1_4 &
CUDA_VISIBLE_DEVICES=4 python src/train.py task=walker-walk amlt=0 seed=5 latent_action_dim=6 act_reconstruct_coef=0.1 inv_consistency_coef=1 exp_name=walker_walk_latent6_alpha1e-1_inv1_1 &
CUDA_VISIBLE_DEVICES=5 python src/train.py task=walker-walk amlt=0 seed=6 latent_action_dim=6 act_reconstruct_coef=0.1 inv_consistency_coef=1 exp_name=walker_walk_latent6_alpha1e-1_inv1_2 &
CUDA_VISIBLE_DEVICES=6 python src/train.py task=walker-walk amlt=0 seed=7 latent_action_dim=6 act_reconstruct_coef=0.1 inv_consistency_coef=1 exp_name=walker_walk_latent6_alpha1e-1_inv1_3 &
CUDA_VISIBLE_DEVICES=7 python src/train.py task=walker-walk amlt=0 seed=8 latent_action_dim=6 act_reconstruct_coef=0.1 inv_consistency_coef=1 exp_name=walker_walk_latent6_alpha1e-1_inv1_4 &

### Experiment 17,22
CUDA_VISIBLE_DEVICES=0 python src/train.py task=walker-walk amlt=0 seed=5 latent_action_dim=2 act_reconstruct_coef=2 inv_consistency_coef=5 exp_name=walker_walk_latent2_alpha2_inv5_1 &
CUDA_VISIBLE_DEVICES=1 python src/train.py task=walker-walk amlt=0 seed=6 latent_action_dim=2 act_reconstruct_coef=2 inv_consistency_coef=5 exp_name=walker_walk_latent2_alpha2_inv5_2 &
CUDA_VISIBLE_DEVICES=2 python src/train.py task=walker-walk amlt=0 seed=7 latent_action_dim=2 act_reconstruct_coef=2 inv_consistency_coef=5 exp_name=walker_walk_latent2_alpha2_inv5_3 &
CUDA_VISIBLE_DEVICES=3 python src/train.py task=walker-walk amlt=0 seed=8 latent_action_dim=2 act_reconstruct_coef=2 inv_consistency_coef=5 exp_name=walker_walk_latent2_alpha2_inv5_4 &
CUDA_VISIBLE_DEVICES=4 python src/train.py task=walker-walk amlt=0 seed=5 latent_action_dim=4 act_reconstruct_coef=2 inv_consistency_coef=5 exp_name=walker_walk_latent4_alpha2_inv5_1 &
CUDA_VISIBLE_DEVICES=5 python src/train.py task=walker-walk amlt=0 seed=6 latent_action_dim=4 act_reconstruct_coef=2 inv_consistency_coef=5 exp_name=walker_walk_latent4_alpha2_inv5_2 &
CUDA_VISIBLE_DEVICES=6 python src/train.py task=walker-walk amlt=0 seed=7 latent_action_dim=4 act_reconstruct_coef=2 inv_consistency_coef=5 exp_name=walker_walk_latent4_alpha2_inv5_3 &
CUDA_VISIBLE_DEVICES=7 python src/train.py task=walker-walk amlt=0 seed=8 latent_action_dim=4 act_reconstruct_coef=2 inv_consistency_coef=5 exp_name=walker_walk_latent4_alpha2_inv5_4 &

### Experiment 6,8
CUDA_VISIBLE_DEVICES=0 python src/train.py task=walker-walk amlt=0 seed=5 latent_action_dim=6 act_reconstruct_coef=1 inv_consistency_coef=0.1 exp_name=walker_walk_latent6_alpha1_inv1e-1_1 &
CUDA_VISIBLE_DEVICES=1 python src/train.py task=walker-walk amlt=0 seed=6 latent_action_dim=6 act_reconstruct_coef=1 inv_consistency_coef=0.1 exp_name=walker_walk_latent6_alpha1_inv1e-1_2 &
CUDA_VISIBLE_DEVICES=2 python src/train.py task=walker-walk amlt=0 seed=7 latent_action_dim=6 act_reconstruct_coef=1 inv_consistency_coef=0.1 exp_name=walker_walk_latent6_alpha1_inv1e-1_3 &
CUDA_VISIBLE_DEVICES=3 python src/train.py task=walker-walk amlt=0 seed=8 latent_action_dim=6 act_reconstruct_coef=1 inv_consistency_coef=0.1 exp_name=walker_walk_latent6_alpha1_inv1e-1_4 &
CUDA_VISIBLE_DEVICES=4 python src/train.py task=walker-walk amlt=0 seed=5 latent_action_dim=6 act_reconstruct_coef=1 inv_consistency_coef=2 exp_name=walker_walk_latent6_alpha1_inv2_1 &
CUDA_VISIBLE_DEVICES=5 python src/train.py task=walker-walk amlt=0 seed=6 latent_action_dim=6 act_reconstruct_coef=1 inv_consistency_coef=2 exp_name=walker_walk_latent6_alpha1_inv2_2 &
CUDA_VISIBLE_DEVICES=6 python src/train.py task=walker-walk amlt=0 seed=7 latent_action_dim=6 act_reconstruct_coef=1 inv_consistency_coef=2 exp_name=walker_walk_latent6_alpha1_inv2_3 &
CUDA_VISIBLE_DEVICES=7 python src/train.py task=walker-walk amlt=0 seed=8 latent_action_dim=6 act_reconstruct_coef=1 inv_consistency_coef=2 exp_name=walker_walk_latent6_alpha1_inv2_4 &

### Experiemnt 12,14

CUDA_VISIBLE_DEVICES=0 python src/train.py task=walker-walk amlt=0 seed=5 latent_action_dim=6 act_reconstruct_coef=2 inv_consistency_coef=1 exp_name=walker_walk_latent6_alpha2_inv1_1 &
CUDA_VISIBLE_DEVICES=1 python src/train.py task=walker-walk amlt=0 seed=6 latent_action_dim=6 act_reconstruct_coef=2 inv_consistency_coef=1 exp_name=walker_walk_latent6_alpha2_inv1_2 &
CUDA_VISIBLE_DEVICES=2 python src/train.py task=walker-walk amlt=0 seed=7 latent_action_dim=6 act_reconstruct_coef=2 inv_consistency_coef=1 exp_name=walker_walk_latent6_alpha2_inv1_3 &
CUDA_VISIBLE_DEVICES=3 python src/train.py task=walker-walk amlt=0 seed=8 latent_action_dim=6 act_reconstruct_coef=2 inv_consistency_coef=1 exp_name=walker_walk_latent6_alpha2_inv1_4 &

CUDA_VISIBLE_DEVICES=0 python src/train.py task=walker-walk amlt=0 seed=5 latent_action_dim=6 act_reconstruct_coef=2 inv_consistency_coef=5 exp_name=walker_walk_latent6_alpha2_inv5_1 &
CUDA_VISIBLE_DEVICES=1 python src/train.py task=walker-walk amlt=0 seed=6 latent_action_dim=6 act_reconstruct_coef=2 inv_consistency_coef=5 exp_name=walker_walk_latent6_alpha2_inv5_2 &
CUDA_VISIBLE_DEVICES=2 python src/train.py task=walker-walk amlt=0 seed=7 latent_action_dim=6 act_reconstruct_coef=2 inv_consistency_coef=5 exp_name=walker_walk_latent6_alpha2_inv5_3 &
CUDA_VISIBLE_DEVICES=3 python src/train.py task=walker-walk amlt=0 seed=8 latent_action_dim=6 act_reconstruct_coef=2 inv_consistency_coef=5 exp_name=walker_walk_latent6_alpha2_inv5_4 &



CUDA_VISIBLE_DEVICES=0 python src/train.py task=dog-walk amlt=0 seed=5 latent_action_dim=20 act_reconstruct_coef=100 inv_consistency_coef=5 exp_name=dog_walk_latent20_alpha100_inv5_1 &
CUDA_VISIBLE_DEVICES=1 python src/train.py task=dog-walk amlt=0 seed=6 latent_action_dim=20 act_reconstruct_coef=100 inv_consistency_coef=5 exp_name=dog_walk_latent20_alpha100_inv5_2 &
CUDA_VISIBLE_DEVICES=2 python src/train.py task=dog-walk amlt=0 seed=7 latent_action_dim=20 act_reconstruct_coef=100 inv_consistency_coef=5 exp_name=dog_walk_latent20_alpha100_inv5_3 &
CUDA_VISIBLE_DEVICES=3 python src/train.py task=dog-walk amlt=0 seed=8 latent_action_dim=20 act_reconstruct_coef=100 inv_consistency_coef=5 exp_name=dog_walk_latent20_alpha100_inv5_4 &
CUDA_VISIBLE_DEVICES=4 python src/train.py task=dog-walk amlt=0 seed=5 latent_action_dim=20 act_reconstruct_coef=20 inv_consistency_coef=5 exp_name=dog_walk_latent20_alpha20_inv5_1 &
CUDA_VISIBLE_DEVICES=5 python src/train.py task=dog-walk amlt=0 seed=6 latent_action_dim=20 act_reconstruct_coef=20 inv_consistency_coef=5 exp_name=dog_walk_latent20_alpha20_inv5_2 &
CUDA_VISIBLE_DEVICES=6 python src/train.py task=dog-walk amlt=0 seed=7 latent_action_dim=20 act_reconstruct_coef=20 inv_consistency_coef=5 exp_name=dog_walk_latent20_alpha20_inv5_3 &
CUDA_VISIBLE_DEVICES=7 python src/train.py task=dog-walk amlt=0 seed=8 latent_action_dim=20 act_reconstruct_coef=20 inv_consistency_coef=5 exp_name=dog_walk_latent20_alpha20_inv5_4 &


CUDA_VISIBLE_DEVICES=0 python src/train.py task=dog-walk amlt=0 seed=5 latent_action_dim=38 act_reconstruct_coef=0 inv_consistency_coef=0 exp_name=dog_walk_1 &
CUDA_VISIBLE_DEVICES=1 python src/train.py task=dog-walk amlt=0 seed=6 latent_action_dim=38 act_reconstruct_coef=0 inv_consistency_coef=0 exp_name=dog_walk_2 &
CUDA_VISIBLE_DEVICES=2 python src/train.py task=dog-walk amlt=0 seed=7 latent_action_dim=38 act_reconstruct_coef=0 inv_consistency_coef=0 exp_name=dog_walk_3 &
CUDA_VISIBLE_DEVICES=3 python src/train.py task=dog-walk amlt=0 seed=8 latent_action_dim=38 act_reconstruct_coef=0 inv_consistency_coef=0 exp_name=dog_walk_4 &


CUDA_VISIBLE_DEVICES=0 python src/train.py task=walker-run amlt=0 seed=1 latent_action_dim=4 act_reconstruct_coef=2 inv_consistency_coef=10 exp_name=walker_run_latent4_alpha2_inv10_1 &
CUDA_VISIBLE_DEVICES=1 python src/train.py task=walker-run amlt=0 seed=2 latent_action_dim=4 act_reconstruct_coef=2 inv_consistency_coef=10 exp_name=walker_run_latent4_alpha2_inv10_2 &
CUDA_VISIBLE_DEVICES=2 python src/train.py task=walker-run amlt=0 seed=3 latent_action_dim=4 act_reconstruct_coef=2 inv_consistency_coef=10 exp_name=walker_run_latent4_alpha2_inv10_3 &
CUDA_VISIBLE_DEVICES=3 python src/train.py task=walker-run amlt=0 seed=4 latent_action_dim=4 act_reconstruct_coef=2 inv_consistency_coef=10 exp_name=walker_run_latent4_alpha2_inv10_4 &
CUDA_VISIBLE_DEVICES=4 python src/train.py task=walker-run amlt=0 seed=1 latent_action_dim=3 act_reconstruct_coef=2 inv_consistency_coef=10 exp_name=walker_run_latent3_alpha2_inv10_1 &
CUDA_VISIBLE_DEVICES=5 python src/train.py task=walker-run amlt=0 seed=2 latent_action_dim=3 act_reconstruct_coef=2 inv_consistency_coef=10 exp_name=walker_run_latent3_alpha2_inv10_2 &
CUDA_VISIBLE_DEVICES=6 python src/train.py task=walker-run amlt=0 seed=3 latent_action_dim=3 act_reconstruct_coef=2 inv_consistency_coef=10 exp_name=walker_run_latent3_alpha2_inv10_3 &
CUDA_VISIBLE_DEVICES=7 python src/train.py task=walker-run amlt=0 seed=4 latent_action_dim=3 act_reconstruct_coef=2 inv_consistency_coef=10 exp_name=walker_run_latent3_alpha2_inv10_4 &

CUDA_VISIBLE_DEVICES=0 python src/train.py task=walker-run amlt=0 seed=5 latent_action_dim=6 act_reconstruct_coef=0 inv_consistency_coef=0 exp_name=walker_run_1 &
CUDA_VISIBLE_DEVICES=1 python src/train.py task=walker-run amlt=0 seed=6 latent_action_dim=6 act_reconstruct_coef=0 inv_consistency_coef=0 exp_name=walker_run_2 &
CUDA_VISIBLE_DEVICES=2 python src/train.py task=walker-run amlt=0 seed=7 latent_action_dim=6 act_reconstruct_coef=0 inv_consistency_coef=0 exp_name=walker_run_3 &
CUDA_VISIBLE_DEVICES=3 python src/train.py task=walker-run amlt=0 seed=8 latent_action_dim=6 act_reconstruct_coef=0 inv_consistency_coef=0 exp_name=walker_run_4 &

[0.1,1,2,5,10] [0.1,1,2,5,10]
#1 (0.1,0.1)
#2 (0.1,1) x
#3 (0.1,2)
#4 (0.1,5)
#5 (0.1,10)

#6  (1,0.1) x
#7  (1,1) x
#8  (1,2) x
#9  (1,5)
#10 (1,10)

#11  (2,0.1)
#12  (2,1) x
#13  (2,2) 
#14  (2,5) x
#15  (2,10)

#16  (5,0.1)
#17  (5,1) x
#18  (5,2)
#19  (5,5)
#20  (5,10)

#21  (10,0.1)
#22  (10,1) x
#23  (10,2)
#24  (10,5)
#25  (10,10)

