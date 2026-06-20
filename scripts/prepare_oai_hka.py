"""
Build the OAI HKA ground-truth table for external validation.

Input : oai_xralign01.txt  (OAI Knee and Full Limb X-Ray Alignment, NDA export)
        oai_xrmeta01.txt    (OAI X-Ray Metadata, NDA export)
Output: oai_hka_groundtruth.csv  with one row per full-limb image (barcode):
        barcode, subj, visit, sex, ageyears,
        hka_side1, hka_side2          (mean OAISYS HKA, signed, 0 = neutral, - = varus)
        nread_side1, nread_side2,     (number of expert reads averaged)
        readsd_side1, readsd_side2    (OAI within-knee reader SD -> the benchmark)
Run:  python prepare_oai_hka.py /path/to/oai_xralign01.txt /path/to/oai_xrmeta01.txt
"""
import sys, pandas as pd, numpy as np

align_path = sys.argv[1] if len(sys.argv) > 1 else "oai_xralign01.txt"
meta_path  = sys.argv[2] if len(sys.argv) > 2 else "oai_xrmeta01.txt"

def load_nda(path):
    d = pd.read_csv(path, sep="\t", skiprows=[1], dtype=str, quoting=3)  # row1 = NDA definitions
    d.columns = [c.strip('"') for c in d.columns]
    for c in d.columns:
        d[c] = d[c].str.strip('"')
    return d

al = load_nda(align_path)
me = load_nda(meta_path)
al["hk"] = pd.to_numeric(al["hkangle"], errors="coerce")
H = al[al["hk"].notna()].copy()                                  # HKA-populated rows only

# keep only Full Limb exams (HKA is only valid there; confirm via metadata examtype)
bc2exam = me.drop_duplicates("barcode").set_index("barcode")["examtype"]
H["examtype"] = H["barcode"].map(bc2exam)
H = H[H["examtype"] == "Full Limb"].copy()

# average multiple expert reads per (barcode, side); keep read count + SD (OAI reader agreement)
agg = (H.groupby(["barcode", "side"])
         .agg(hka=("hk", "mean"), nread=("hk", "size"), readsd=("hk", "std"),
              subj=("src_subject_id", "first"), visit=("visit", "first"),
              sex=("sex", "first"), ageyears=("ageyears", "first"))
         .reset_index())

# pivot to one row per barcode (full-limb image) with both knees
def pivot(col):
    p = agg.pivot_table(index="barcode", columns="side", values=col, aggfunc="first")
    p.columns = [f"{col}_side{c}" for c in p.columns]
    return p
wide = (agg.drop_duplicates("barcode").set_index("barcode")[["subj", "visit", "sex", "ageyears"]]
          .join([pivot("hka"), pivot("nread"), pivot("readsd")]).reset_index())

wide.to_csv("oai_hka_groundtruth.csv", index=False)
print(f"Full-limb images: {len(wide)} | both knees: {wide[['hka_side1','hka_side2']].notna().all(1).sum()}")
print(f"OAI within-knee reader SD (mean): "
      f"{pd.concat([wide['readsd_side1'], wide['readsd_side2']]).dropna().mean():.3f} deg")
print("Wrote oai_hka_groundtruth.csv")
