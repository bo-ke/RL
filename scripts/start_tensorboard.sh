
nohup tensorboard --samples_per_plugin scalars=100000 --logdir ./logs --host $(hostname -i) --port 8040 > tensorboard.log 2>&1 &