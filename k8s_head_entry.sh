#!/bin/bash
HEAD_NODE=$(kubectl get pods --no-headers -o custom-columns=":metadata.name" | grep head)
kubectl exec --stdin --tty $HEAD_NODE -- \
    /bin/bash -c "uv run python scripts/training/cleanrl/train_moaim_ppo.py config/simple_tag/algatross_k8s.yml"
