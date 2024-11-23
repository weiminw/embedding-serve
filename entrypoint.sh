#!/bin/bash
set -e
if [[ $# -eq 0 ]]; then
  exec "/bin/bash"
else
  source /workspace/heliumos-env/bin/activate
  python -m bixi.embeddings.serve $@
fi