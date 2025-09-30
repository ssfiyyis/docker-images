#!/bin/bash
docker build --network host -t vh0027635/nemo-and-run:0.0.1 -f Dockerfile.pretrain .
docker run --name nemo-test1 -v $HOME/.cache/huggingface:/root/.cache/huggingface -itd --gpus=all --ipc=host vh0027635/nemo-and-run:0.0.1
docker exec -it $container_id /bin/bash
