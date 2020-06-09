#!/bin/bash

cred_helper=$(git config credential.helper)

if [ -z "$cred_helper" ]; then
    echo "Setting 'git config --global credential.helper store'..."
    git config --global credential.helper store
fi

mkdir /kaggle/tmp
mkdir /kaggle/working
