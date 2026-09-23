#!/usr/bin/env bash
set -e
cd /mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline
PY=/home/yujun/.cache/ai_iaps_pilot_venv/bin/python3
K=/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/Decoding/full_cohort_raw_pipeline_28/resources/kastner
OUT=roi_decoding_random10fold_100rep
SUBJECTS="1 2 4 5 6 7 9 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 33 34"

for s in $SUBJECTS; do
  pad=$(printf "%02d" $s)
  out_dir="$OUT/sub-$pad"
  if [ -f "$out_dir/subject_results.csv" ]; then
    echo "sub-$pad: skipped_existing"
    continue
  fi
  $PY code/decode_roi_random10fold.py \
    --subject Sub$pad \
    --branch mni_res_native \
    --smoothing-mm 8 \
    --input-root glmsingle/sub-$pad \
    --atlas $K/kastner.nii.gz --atlas-labels $K/kastner.nii.txt \
    --output "$out_dir" \
    --n-repeats 100 \
    --n-jobs 20 \
    > logs/decode_roi_random10fold_100rep_sub$pad.log 2>&1
  status=$?
  if [ $status -eq 0 ]; then
    echo "sub-$pad: pass"
  else
    echo "sub-$pad: fail (see logs/decode_roi_random10fold_100rep_sub$pad.log)"
  fi
done
echo RANDOM10FOLD_100REP_WAVE_COMPLETE
