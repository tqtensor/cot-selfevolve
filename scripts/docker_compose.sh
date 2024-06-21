#!/bin/bash

mkdir -p src/docker/data
tar xzf src/docker/chroma-data.tar.gz -C src/docker/data
docker compose -f src/docker/docker-compose.yaml up -d
