#!/bin/bash
HEAD_NODE=$(kubectl get pods --no-headers -o custom-columns=":metadata.name" | grep head)
kubectl exec --stdin --tty $HEAD_NODE -- \
    /bin/bash -c "${1}"
