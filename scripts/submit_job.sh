#!/bin/bash

# Get the latest commit hash
COMMIT_HASH=$(git rev-parse HEAD)

# Submit the job with the latest commit hash
.venv/bin/python scripts/deploy.py --action submit --commit-hash $COMMIT_HASH
