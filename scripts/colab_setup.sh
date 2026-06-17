#!/usr/bin/env bash
set -euo pipefail

apt-get install -y -q espeak-ng libespeak-ng-dev

pip install -q \
    'transformers==5.9.0' \
    'datasets==3.6.0' \
    'soundfile==0.13.1' \
    'librosa==0.11.0' \
    'scipy==1.17.1' \
    'accelerate==1.13.0' \
    'phonemizer==3.3.0'

python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.version.cuda)"
