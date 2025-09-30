#!/bin/bash
docker build --network host -t vh0027635/nemo-and-run:0.1.0 -f Dockerfile.pretrain .
docker run --name nemo-test1 -v $HOME/.cache/huggingface:/root/.cache/huggingface -itd --gpus=all --ipc=host vh0027635/nemo-and-run:0.1.0
# To get into container
docker exec -it $container_id /bin/bash
# To run workload from outside container
docker run -v /home/ubuntu/src/docker-images:/workspace --rm --gpus=all --ipc=host --entrypoint bash vh0027635/nemo-and-run:0.1.0 -lc 'source /opt/venv/bin/activate && python dummy_train.py'
