#!/usr/bin/env bash
set -euo pipefail

probe=/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/full_cohort_raw_pipeline_28/runtime/smoke/docker_n_mount_20260729
image=nipreps/fmriprep@sha256:4e5cfd99f6d80a9ef10a87929f8e74e4caf9dc108b49551eb23c775e61cd16f7

test -s "$probe/host_input.txt"
test ! -e "$probe/wsl_container_output.txt"

docker run --rm --network none \
  -v "$probe:/probe" \
  --entrypoint python \
  "$image" \
  -c 'from pathlib import Path; p=Path("/probe/host_input.txt"); q=Path("/probe/wsl_container_output.txt"); x=p.read_bytes(); assert x; q.write_bytes(x); print(len(x))'

cmp "$probe/host_input.txt" "$probe/wsl_container_output.txt"
sha256sum "$probe/host_input.txt" "$probe/wsl_container_output.txt"
