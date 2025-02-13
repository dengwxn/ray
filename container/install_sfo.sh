#!/bin/bash

sudo apt update
sudo apt install python3.10-dev python3.10-venv curl wget -y

cd
git clone https://github.com/dengwxn/ray.git Ray-SFO

cd ~/Ray-SFO
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel

~/Ray-SFO/ci/env/install-bazel.sh --user
export PATH="$PATH:$HOME/bin"
bazel --version

cd
wget https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh
chmod +x install.sh
./install.sh
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
rm install.sh
nvm install 14
nvm use 14

cd ~/Ray-SFO/python/ray/dashboard/client
npm ci
npm run build

cd ~/Ray-SFO/python
pip install -r requirements.txt
pip install -e . --verbose
pip install pytest torch
pip install -c requirements_compiled.txt -r requirements/lint-requirements.txt
pip install --upgrade typing-extensions
