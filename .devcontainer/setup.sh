#!/bin/bash

cred_helper=$(git config credential.helper)

if [ -z "$cred_helper" ]; then
    echo "Setting 'git config --global credential.helper store'..."
    git config --global credential.helper store
fi

mkdir /kaggle/tmp
mkdir /kaggle/working

pip install --upgrade comet_ml

# Set up a link to the API key to root's home.
ln -s /workspaces/kaggle/kaggle.json /root/.kaggle/kaggle.json
chmod 600 /root/.kaggle/kaggle.json
