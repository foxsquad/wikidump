#!/bin/bash

mkdir -p data

read -r HASH < "$(dirname $0)/TORRENT_HASH"

aria2c -T "data/${HASH}.torrent" -d data --file-allocation=falloc -V
