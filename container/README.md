# README

## Build
```bash
cd ray
docker build -t ray-blacklist .
docker images | grep ray-blacklist
docker tag ray-blacklist:latest dengwxn/ray-blacklist:latest
docker push dengwxn/ray-blacklist:latest
```

## Run
```bash
docker run -d --name ray-blacklist-run \
  --network=host \
  --shm-size=128gb \
  --gpus all \
  --cap-add SYS_PTRACE \
  -v $(pwd):/app \
  -it ray-blacklist
docker exec -it ray-blacklist-run zsh
```

## Clean
```bash
docker stop ray-blacklist-run
docker rm ray-blacklist-run
```

## Install
```bash
cd /app/container
./install_container.sh
```
