export NCCL_IB_DISABLE=1      # 禁用 InfiniBand 通信
export NCCL_P2P_DISABLE=1     # 禁用点对点通信
export NCCL_SHM_DISABLE=1     # 禁用共享内存通道

deepspeed --num_gpus=8 --master_port=30000 \
    /opt/tiger/dyf/dyf/train/train.py \
    --deepspeed ds_config.json \