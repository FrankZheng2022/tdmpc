GAME=$1
TASK=$2
SEED=$3
AMLT=$4
LATENT_ACT_DIM=$5
INV_COEFF=$6
ACT_COEFF=$7

cd dmc2gym &&  pip install -e . && cd ../
pip install numpy==1.20.3 

python src/train.py task=${GAME}-${TASK} \
                    seed=${SEED}\
                    latent_action_dim=${LATENT_ACT_DIM} \
                    inv_consistency_coef=${INV_COEFF}\
                    act_reconstruct_coef=${ACT_COEFF}\
                    exp_name=${GAME}_${TASK}_latent${LATENT_ACT_DIM}_inv${INV_COEFF}_act${ACT_COEFF}_s${SEED}\
                    amlt=${AMLT}

