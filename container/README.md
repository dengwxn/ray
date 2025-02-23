# README

## Build
```bash
cd ray
docker build -t ray-sea2bel-unify .
docker images | grep ray-sea2bel-unify
docker tag ray-sea2bel-unify:latest dengwxn/ray-sea2bel-unify:latest
docker push dengwxn/ray-sea2bel-unify:latest
```

## Run
```bash
docker run -d --name ray-sea2bel-unify-run \
  --shm-size=128gb \
  --gpus all \
  --cap-add SYS_PTRACE \
  -v $(pwd):/app \
  -it ray-sea2bel-unify
docker exec -it ray-sea2bel-unify-run zsh
```

## Clean
```bash
docker stop ray-sea2bel-unify-run
docker rm ray-sea2bel-unify-run
```

## Install
```bash
cd /app/container
./install_container.sh
```
