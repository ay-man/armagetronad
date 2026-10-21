#!/bin/bash
# EC2 user data (Ubuntu 24.04 AMI): installs Docker, fetches this repository and starts the
# bot servers from docker-compose.yml. Edit REPO/BRANCH; for a private repository put a
# read-only token in the URL (https://<token>@github.com/<org>/armagetronad.git).
set -euxo pipefail
REPO=https://github.com/ay-man/armagetronad.git
BRANCH=rl/fort-learning

apt-get update
apt-get install -y ca-certificates curl git
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

git clone --depth 1 --branch "$BRANCH" "$REPO" /opt/armagetronad
cd /opt/armagetronad
docker compose -f rl/deploy/docker-compose.yml up -d --build
