#!/bin/bash
# What the sandbox isolates, checked from inside.
echo "kernel: $(uname -sr)"; head -2 /etc/os-release
echo "user: $(id)"; echo "cpus: $(nproc) mem: $(free -g | awk '/Mem/{print $2}')G"
for b in python3 git curl uv; do printf "%s: " $b; command -v $b || echo missing; done
echo "--- network"
printf "ollama (allowed): "; curl -s -m 5 http://host.docker.internal:11434/api/version || echo BLOCKED
printf "\nollaya :11435 (allowed, tunnel to the Mac): "; curl -s -m 5 -o /dev/null -w "%{http_code}" http://host.docker.internal:11435/ || echo BLOCKED
printf "\notlp (allowed): "; curl -s -m 5 -o /dev/null -w "%{http_code}" http://host.docker.internal:4318/v1/traces -X POST -H 'Content-Type: application/json' -d '{}' || echo BLOCKED
printf "\nexample.com (should be blocked): "; curl -s -m 5 -o /dev/null -w "%{http_code}" https://example.com || echo BLOCKED
printf "\nphoenix :6006 (not allowed): "; curl -s -m 5 -o /dev/null -w "%{http_code}" http://host.docker.internal:6006 || echo BLOCKED
printf "\npypi (closed after setup): "; curl -s -m 5 -o /dev/null -w "%{http_code}" https://pypi.org/simple/ || echo BLOCKED
printf "\nhost files outside the workspace: "; ls /c/Users 2>&1 | head -1
