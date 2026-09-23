#!/usr/bin/env python3
"""Put each MRIcroGL snapshot on its own PowerPoint slide, one image per slide, no text.

Coordinates (per user instruction):
  Pleasant vs Neutral (natural + AI):   x=-17, y=19, z=-2
  Unpleasant vs Neutral (natural + AI): x=50, y=-76, z=-13
Colormap and scale match the earlier composite figure: '4hot', minmax(3, 11).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.util import Inches

PANELS = Path("/mnt/c/Users/YUJUNC~1/AppData/Local/Temp/claude/C--/794a7e21-d3b3-4d39-be02-428ce42a9c00/scratchpad/panels_req4")
OUT_PPTX = Path("/mnt/n/Experimental_Data/yujunchen/projects/AI_IAPS/new_pipeline/categorical_glm/group_n29/figure_mricrogl/mricrogl_snapshots_scalemax7.pptx")

ROWS = ["natural_pleasant", "ai_pleasant", "natural_unpleasant", "ai_unpleasant"]
VIEWS = ["sagittal", "coronal", "axial"]


def crop_to_content(img: Image.Image, pad: int = 15) -> Image.Image:
    arr = np.asarray(img.convert("L"))
    rows = np.where(arr.max(axis=1) > 10)[0]
    cols = np.where(arr.max(axis=0) > 10)[0]
    if rows.size == 0 or cols.size == 0:
        return img
    top, bottom = max(rows[0] - pad, 0), min(rows[-1] + pad, arr.shape[0])
    left, right = max(cols[0] - pad, 0), min(cols[-1] + pad, arr.shape[1])
    return img.crop((left, top, right, bottom))


prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]

tmp_dir = PANELS / "_cropped_for_pptx"
tmp_dir.mkdir(exist_ok=True)

for row in ROWS:
    for view in VIEWS:
        src = PANELS / f"{row}_{view}.png"
        img = crop_to_content(Image.open(src))
        cropped_path = tmp_dir / f"{row}_{view}.png"
        img.save(cropped_path)

        slide = prs.slides.add_slide(blank_layout)
        img_w, img_h = img.size
        aspect = img_w / img_h
        max_w, max_h = Inches(11.0), Inches(6.8)
        if max_w / aspect <= max_h:
            disp_w = max_w
            disp_h = max_w / aspect
        else:
            disp_h = max_h
            disp_w = max_h * aspect
        left = (prs.slide_width - disp_w) / 2
        top = (prs.slide_height - disp_h) / 2
        slide.shapes.add_picture(str(cropped_path), left, top, width=disp_w, height=disp_h)

OUT_PPTX.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT_PPTX)
print(f"saved {OUT_PPTX} with {len(ROWS) * len(VIEWS)} slides")
