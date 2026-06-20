"""
Turn the full-limb relative paths into real s3 links using the package metadata,
so they can be downloaded with:  downloadcmd -dp 1247031 -t fulllimb_s3links.txt -d ./oai_fulllimb

Usage:
    python build_fulllimb_s3links.py filtered/image03.txt package_file_metadata_1247031.txt.gz

Outputs:
    fulllimb_s3links.txt        -> one s3:// link per full-limb .tar.gz (feed to downloadcmd -t)
    fulllimb_barcode_s3_map.csv  -> barcode, image_file, s3_url  (for the notebook to match later)
"""
import sys, gzip, csv

img03 = sys.argv[1] if len(sys.argv) > 1 else "filtered/image03.txt"
meta  = sys.argv[2] if len(sys.argv) > 2 else "package_file_metadata_1247031.txt.gz"

# 1) read the filtered full-limb structure file: image_file (.tar.gz) -> barcode
with open(img03) as f:
    cols = [c.strip().strip('"') for c in f.readline().rstrip("\n").split("\t")]
    f.readline()                                              # NDA definitions row
    ai, ifi = cols.index("accession_number"), cols.index("image_file")
    want = {}                                                 # image_file -> barcode
    for line in f:
        fld = line.rstrip("\n").split("\t")
        if len(fld) > max(ai, ifi):
            want[fld[ifi].strip().strip('"')] = fld[ai].strip().strip('"')
print(f"full-limb archives to locate: {len(want)}")

# 2) stream the gz metadata: DOWNLOAD_ALIAS -> NDA_S3_URL for our files only
rows = []
opener = gzip.open if meta.endswith(".gz") else open
with opener(meta, "rt") as f:
    r = csv.reader(f); h = next(r)
    da, su = h.index("DOWNLOAD_ALIAS"), h.index("NDA_S3_URL")
    for row in r:
        if len(row) > max(da, su) and row[da] in want:
            rows.append((want[row[da]], row[da], row[su]))
print(f"matched s3 links: {len(rows)}  (distinct barcodes: {len({b for b,_,_ in rows})})")

# 3) write the s3 list + a barcode->s3 map
with open("../data/hka/fulllimb_s3links.txt", "w") as out:
    for _, _, s3 in rows:
        out.write(s3 + "\n")
with open("../data/hka/fulllimb_barcode_s3_map.csv", "w", newline="") as out:
    w = csv.writer(out); w.writerow(["barcode", "image_file", "s3_url"])
    w.writerows(rows)
print("wrote fulllimb_s3links.txt + fulllimb_barcode_s3_map.csv")
miss = set(want.values()) - {b for b, _, _ in rows}
if miss:
    print(f"WARNING: {len(miss)} barcodes had no s3 match (e.g. {list(miss)[:3]})")
