"""
Extract each full-limb .tar.gz (which contains a single DICOM named '001')
and rename it to <barcode>.dcm so the notebook can match images to HKA by stem.

Usage:
    python extract_oai_dicoms.py oai_fulllimb fulllimb_barcode_s3_map.csv oai_dicoms
Output:
    oai_dicoms/<barcode>.dcm   (one per archive)  ->  set OAI_IMAGE_DIR to this folder
"""
import sys, os, glob, tarfile, shutil, csv

src = sys.argv[1] if len(sys.argv) > 1 else "oai_fulllimb"
mapf = sys.argv[2] if len(sys.argv) > 2 else "fulllimb_barcode_s3_map.csv"
dst = sys.argv[3] if len(sys.argv) > 3 else "oai_dicoms"
os.makedirs(dst, exist_ok=True)

# archive basename -> barcode
b2bc = {}
with open(mapf) as f:
    for row in csv.DictReader(f):
        b2bc[os.path.basename(row["image_file"])] = row["barcode"]
print(f"barcode map entries: {len(b2bc)}")

archives = glob.glob(os.path.join(src, "**", "*.tar.gz"), recursive=True)
print(f"archives found: {len(archives)}")
done = miss = multi = 0
for a in archives:
    bc = b2bc.get(os.path.basename(a))
    if not bc:
        miss += 1; continue
    with tarfile.open(a) as t:
        files = [m for m in t.getmembers() if m.isfile()]
        if not files:
            continue
        if len(files) > 1:
            multi += 1                       # full-limb should be single-image; note if not
        with t.extractfile(files[0]) as fin, open(os.path.join(dst, f"{bc}.dcm"), "wb") as fout:
            shutil.copyfileobj(fin, fout)
    done += 1

print(f"extracted {done} DICOMs -> {dst}/<barcode>.dcm")
if miss:  print(f"  {miss} archives had no barcode in the map")
if multi: print(f"  note: {multi} archives held >1 file; used the first")
