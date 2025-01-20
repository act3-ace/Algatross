#!/bin/bash -e
NEXT_RELEASE="$1"

# Update Version file
echo "${NEXT_RELEASE}" >VERSION

# Export uv lock file if lock file exists
if [ -f uv.lock ]; then
    uv export --format requirements-txt -o requirements.txt --no-dev --no-group lint --no-group test
    uv export --format requirements-txt -o requirements-dev.txt --only-group dev --only-group test --only-group lint
    uv export --format requirements-txt -o requirements-docs.txt --only-group docs
fi

# ./scripts/pipeline/update_version_list.py
scripts/docs/auto_update_news.py -vvv
