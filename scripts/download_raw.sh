#!/usr/bin/env bash
# Stage 1: download the public ANN datasets into data/raw/<name>/ (one directory each; files
# are untrusted data and are only parsed by numpy/h5py, never executed). Resumable (-C -).
set -euo pipefail
cd "$(dirname "$0")/.."
get() {  # url dest [range]
  local url=$1 dest=$2 range=${3:-}
  if [ -n "$range" ]; then
    local want=$(( ${range#*-} + 1 ))
    if [ -f "$dest" ] && [ "$(stat -c %s "$dest")" -eq "$want" ]; then echo "have $dest"; return; fi
    curl -sS --fail --retry 5 --retry-delay 10 -r "$range" -o "$dest" "$url"
  else
    curl -sS --fail --retry 5 --retry-delay 10 -C - -o "$dest" "$url" || [ $? -eq 33 ]
  fi
  echo "got $dest $(stat -c %s "$dest") bytes"
}
get ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz data/raw/sift/sift.tar.gz
get http://ann-benchmarks.com/glove-100-angular.hdf5 data/raw/glove/glove-100-angular.hdf5
get https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin data/raw/deep/query.public.10K.fbin
# first 10M vectors of Deep1B: 8-byte header (n, d as int32) + 10M x 96 float32
get https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin data/raw/deep/base.10M.partial.fbin 0-3840000007
get ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz data/raw/gist/gist.tar.gz
for d in sift gist; do (cd data/raw/$d && tar xzf $d.tar.gz && echo "extracted $d"); done
echo DOWNLOADS_DONE
