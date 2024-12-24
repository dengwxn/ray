# README

## Build
```bash
cd ray
docker build -t ray-waterfront .
docker images | grep ray-waterfront
docker tag ray-waterfront:latest dengwxn/ray-waterfront:latest
docker push dengwxn/ray-waterfront:latest
```

## Run
```bash
docker run -d --name ray-waterfront-dev \
  --shm-size=128gb \
  --gpus all \
  --cap-add SYS_PTRACE \
  -v $(pwd):/app \
  -it ray-waterfront
docker exec -it ray-waterfront-dev zsh
```

## Clean
```bash
docker stop ray-waterfront-dev
docker rm ray-waterfront-dev
```

## Install
```bash
cd ray/dev
./install.sh
```
