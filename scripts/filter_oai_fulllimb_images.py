"""
Filter image03.txt down to the full-limb radiographs, preserving NDA format,
so it can be fed to:  downloadcmd -dp <PACKAGE_ID> -ds fulllimb_image03.txt -d ./oai_fulllimb

Usage:
    python filter_oai_fulllimb_images.py image03.txt oai_hka_groundtruth.csv
Outputs:
    fulllimb_image03.txt       -> filtered data-structure file (header + defs + full-limb rows) for -ds
    fulllimb_barcode_map.csv    -> barcode, image_file  (for the notebook to match images to HKA)
"""
import sys, pandas as pd
from collections import Counter

img_path = sys.argv[1] if len(sys.argv) > 1 else "image03.txt"
gt_path  = sys.argv[2] if len(sys.argv) > 2 else "oai_hka_groundtruth.csv"

barcodes = set(pd.read_csv(gt_path, dtype=str)["barcode"].str.strip())
print(f"full-limb barcodes to find: {len(barcodes)}")

with open(img_path) as f:
    header = f.readline(); defs = f.readline()                       # NDA: row0 headers, row1 definitions
    cols = [c.strip().strip('"') for c in header.rstrip("\n").split("\t")]
    ai  = cols.index("accession_number")
    imo = cols.index("image_modality")
    ids = cols.index("image_description")
    ifl = cols.index("image_file")

    out = open("../data/hka/fulllimb_image03.txt", "w"); out.write(header); out.write(defs)
    bmap = open("../data/hka/fulllimb_barcode_map.csv", "w"); bmap.write("barcode,image_file\n")
    kept = total = 0; seen = set(); mod = Counter(); desc = Counter(); sample = []
    for line in f:
        total += 1
        fld = line.rstrip("\n").split("\t")
        if len(fld) <= ai:
            continue
        acc = fld[ai].strip().strip('"')
        if acc in barcodes:
            out.write(line); kept += 1; seen.add(acc)
            imgf = fld[ifl].strip().strip('"')
            bmap.write(f"{acc},{imgf}\n")
            mod[fld[imo].strip().strip('"')] += 1
            desc[fld[ids].strip().strip('"')] += 1
            if len(sample) < 3: sample.append(imgf)
    out.close(); bmap.close()

print(f"scanned {total} data rows; kept {kept} full-limb rows ({len(seen)} distinct barcodes)")
print("modality      :", dict(mod))
print("description   :", dict(desc.most_common(5)))
print("sample image_file:", sample)
print("\nwrote fulllimb_image03.txt  +  fulllimb_barcode_map.csv")
