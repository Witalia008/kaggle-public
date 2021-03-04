#!/bin/bash

# Set up a link to the API key to root's home.
mkdir /root/.kaggle
ln -s /workspaces/kaggle/kaggle.json /root/.kaggle/kaggle.json
chmod 600 /root/.kaggle/kaggle.json
