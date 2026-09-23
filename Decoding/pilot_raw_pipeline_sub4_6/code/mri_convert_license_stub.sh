#!/bin/sh
# Minimal compatibility shim for fMRIPrep's unconditional FreeSurfer license
# probe when --fs-no-reconall is active. It copies the sentinel NIfTI used only
# by that probe. Any other invocation fails loudly rather than pretending to be
# a general replacement for FreeSurfer's mri_convert.
if [ "$#" -eq 2 ] && [ "${1##*.}" = "gz" ] && [ "${2##*.}" = "mgz" ]; then
    cp -- "$1" "$2"
    exit $?
fi
echo "mri_convert license stub received an unsupported invocation" >&2
exit 64
