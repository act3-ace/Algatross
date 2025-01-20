#!/bin/bash
for line in $(kubectl get pods --no-headers -o custom-columns=":metadata.name"); do kubectl logs $line; done
