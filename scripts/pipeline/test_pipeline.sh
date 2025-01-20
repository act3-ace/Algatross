#!/bin/bash
IFS=':' read -ra auth <<<$(env | grep ACT3_SECRETS_GITLAB | tr -d "\n" | sed "s/ACT3_SECRETS_GITLAB=//")

ci_user=$(echo ${auth[0]} | tr -d "\n")
ci_password=$(echo ${auth[1]} | tr -d "\n")

CI_REGISTRY_PASSWORD=$ci_password dagger call lint --git-auth-user="${ci_user}" --git-auth-token=env:CI_REGISTRY_PASSWORD
dagger call test
dagger call docs
CI_REGISTRY_PASSWORD=$ci_password CI_JOB_TOKEN=$ci_password dagger call test-pipeline --ci-registry-user="${ci_user}" --ci-registry-password=env:CI_REGISTRY_PASSWORD --ci-user="${ci_user}" --ci-job-token=env:CI_JOB_TOKEN --git-auth-user="${ci_user}" --git-auth-token=env:CI_REGISTRY_PASSWORD
