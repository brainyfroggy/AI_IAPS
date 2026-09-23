#!/usr/bin/env python3
"""Run decode_roi_random10fold.py's exact CV/output logic against an arbitrary
GLMsingle HDF5 filename, by monkeypatching its imported load_glmsingle reference
before calling its main(). decode_roi_random10fold.py's load_glmsingle (imported
from decode_roi_singletrial) hardcodes TYPED_FITHRF_GLMDENOISE_RR.hdf5, so this is
the only way to point it at e.g. TYPEC_FITHRF_GLMDENOISE.hdf5 without touching the
frozen/validated script.

Usage matches decode_roi_random10fold.py's CLI exactly, plus --hdf5-name.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import decode_roi_random10fold as r10f  # noqa: E402
from glmsingle_typeb_erp_pilot import load_glmsingle_variant_full  # noqa: E402


def main() -> None:
    # Pull --hdf5-name out of argv before decode_roi_random10fold's own
    # argparse sees it (it doesn't know that flag).
    argv = sys.argv[1:]
    if "--hdf5-name" not in argv:
        raise RuntimeError("--hdf5-name is required")
    idx = argv.index("--hdf5-name")
    hdf5_name = argv[idx + 1]
    del argv[idx:idx + 2]
    sys.argv = [sys.argv[0]] + argv

    def patched_loader(root, atlas, labels, smoothing):
        return load_glmsingle_variant_full(root, atlas, labels, smoothing, hdf5_name)

    r10f.load_glmsingle = patched_loader
    r10f.main()


if __name__ == "__main__":
    main()
