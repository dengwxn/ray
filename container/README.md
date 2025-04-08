# README

## Build
```bash
cd ray
docker build -t ray-bridge .
docker images | grep ray-bridge
docker tag ray-bridge:latest dengwxn/ray-bridge:latest
docker push dengwxn/ray-bridge:latest

cd ray
docker build -t ray-shunfeng .
docker images | grep ray-shunfeng
docker tag ray-shunfeng:latest dengwxn/ray-shunfeng:latest
docker push dengwxn/ray-shunfeng:latest
```

## Run
```bash
docker run -d --name ray-bridge-run \
  --network=host \
  --shm-size=128gb \
  --gpus all \
  --cap-add SYS_PTRACE \
  -v $(pwd):/app \
  -it ray-bridge
docker exec -it ray-bridge-run zsh

docker run -d --name ray-shunfeng-run \
  --network=host \
  --shm-size=128gb \
  --gpus all \
  --cap-add SYS_PTRACE \
  -v $(pwd):/app \
  -it ray-shunfeng
docker exec -it ray-shunfeng-run zsh
```

## Clean
```bash
docker stop ray-bridge-run
docker rm ray-bridge-run

docker stop ray-shunfeng-run
docker rm ray-shunfeng-run
```

## Install
```bash
cd /app/container
./install_container.sh
```
