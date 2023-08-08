#!/bin/bash

# wgetコマンドが存在するかチェック
if ! type "wget" > /dev/null; then
  echo "Error: wget is not installed."
  exit 1
fi

# CUDAリポジトリの設定ファイルをダウンロード
if ! wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin; then
  echo "Error: Failed to download cuda-wsl-ubuntu.pin"
  exit 1
fi

# CUDAリポジトリの設定ファイルを移動
if ! sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600; then
  echo "Error: Failed to move cuda-wsl-ubuntu.pin"
  exit 1
fi

# CUDAリポジトリのパッケージをダウンロード
if ! wget https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/cuda-repo-wsl-ubuntu-11-4-local_11.4.0-1_amd64.deb; then
  echo "Error: Failed to download cuda-repo-wsl-ubuntu-11-4-local_11.4.0-1_amd64.deb"
  exit 1
fi

# CUDAリポジトリのパッケージをインストール
if ! sudo dpkg -i cuda-repo-wsl-ubuntu-11-4-local_11.4.0-1_amd64.deb; then
  echo "Error: Failed to install cuda-repo-wsl-ubuntu-11-4-local_11.4.0-1_amd64.deb"
  exit 1
fi

# CUDAリポジトリの公開鍵を追加
if ! sudo apt-key add /var/cuda-repo-wsl-ubuntu-11-4-local/7fa2af80.pub; then
  echo "Error: Failed to add CUDA repository key."
  exit 1
fi

# パッケージリストの更新
if ! sudo apt-get update; then
  echo "Error: Failed to update package lists."
  exit 1
fi
