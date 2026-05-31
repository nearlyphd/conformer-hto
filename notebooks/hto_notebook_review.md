# Review: `hto_correction_angles_kfolds.ipynb`

Global one-stage Conformer for surgical-landmark detection on long-leg radiographs, with 5-fold cross-validation and a downstream high-tibial-osteotomy (HTO) correction-angle analysis.

---

## 1. Executive summary

The notebook is well-organised and the methodology is fundamentally sound: letterbox preprocessing, heatmap regression with a masked MSE loss, PCK reported at four thresholds, 5-fold cross-validation, and — importantly — a clinically meaningful downstream endpoint (the Miniaci correction angle α) rather than pixel error alone. On the held-out test set the model is excellent: mean absolute angular error **0.47°** (max 0.99°) across 11 hemisphere cases, which is well inside the ~1° tolerance that matters for HTO planning.

The headline caveat is the dataset size: **54 radiographs total** (43 train / ~11 val per fold; 6 images / 11 hemispheres in the final test set). At that scale a single image dominates any fold's metrics, and that is exactly what produced the performance outlier you asked about.

**The outlier is Fold 3.** Its best-checkpoint keypoint MSE is 32.28 px² versus 4.0–6.6 px² for every other fold, and its correction-angle error distribution has a maximum of **9.878°** and a standard deviation of 2.025° — roughly 5–10× the other folds — while its MAE (0.959°) and PCK@0.01 (98.5%) barely move. That signature is diagnostic: it is not a uniformly worse model but **one or two grossly mislocalised landmarks on a single validation image, amplified into a large angular error by the geometry of the angle computation.**

---

## 2. What the notebook does

The pipeline has four stages. First, each radiograph is letterbox-resized to 768×768, recording padding offsets so predictions can be mapped back to original coordinates. Second, COCO annotations are mapped to a 12-channel Gaussian heatmap target (six landmarks per leg: femoral head, knee inner/outer, osteotomy point, ankle inner/outer), with a left/right hemisphere split based on each annotation's bounding-box centre relative to the image midline. Third, a `small_p16` Conformer is trained for 2,000 epochs with AdamW, cosine annealing, gradient clipping, and a sigmoid+MSE heatmap loss masked to visible keypoints; checkpoints are saved on best validation MSE. Finally, predicted landmarks are fed through the Miniaci geometry (Fujisawa point at 62.5% of the tibial plateau, target-at-ankle intersection, and the correction angle α via `atan2` of the two vectors anchored at the osteotomy hinge), and angular error against ground truth is reported.

The 5-fold CV repeats this from a fresh model per fold and additionally reloads each fold's best-MSE checkpoint to compute correction-angle statistics on that fold's validation split.

---

## 3. Code quality

The code is clean, readable, and well-documented with sensible docstrings, and the execution counts (1–16, sequential) show the notebook was run top-to-bottom in one clean session on an H100. The geometry utilities are factored out and shared between the CV loop and the final evaluation, which avoids drift between the two. Within that solid baseline, several issues are worth addressing — roughly in priority order.

**Per-fold metrics are selected on the same data they are scored on.** Each fold saves its checkpoint on best validation MSE and then reports MSE/PCK/angle on that same validation fold. This introduces a mild optimistic bias (selection-on-the-evaluation-set). It does not invalidate the CV, but the per-fold numbers should be understood as slightly favourable, and the angle statistics are taken from the best-*MSE* epoch, which is not necessarily the best-*angle* epoch — two selection criteria are being conflated.

**The aggregate is dominated by one fold, yet reported as mean ± std.** "Keypoint MSE 10.32 ± 11.02 px²" and "Angle Max 2.900 ± 3.498°" both have a standard deviation larger than the mean — a clear sign the summary is being driven by Fold 3 rather than describing a typical fold. A mean ± std here is statistically misleading.

**MSE in px² is the wrong headline metric for keypoints.** It is a mean of squared pixel errors, so it is extremely sensitive to a single gross outlier: one landmark misplaced by ~55–80 px on one image adds enough to push a fold's mean from ~5 to ~32 px² on its own. Median per-keypoint Euclidean error (or RMSE in px) plus PCK would be far more interpretable and robust, and would show Fold 3 as only mildly worse rather than catastrophically so.

**Reproducibility is incomplete.** `set_seed(42)` is called once, globally, before the CV loop, and `cudnn.deterministic=True` is set — but the `DataLoader`s use `num_workers=4` with augmentation built on Python's `random` and `np.random` and **no `worker_init_fn` and no `generator=`**. Data ordering and augmentation are therefore not actually reproducible, and folds are not seeded independently. This matters directly for the outlier: the run cannot be reproduced exactly, so some of Fold 3's behaviour cannot be cleanly separated from run-to-run noise.

**The fold evaluation discards exactly the information needed to debug the outlier.** `evaluate_fold_angles` returns only aggregate `{mae, std, max, n}`. There is no per-image error and no filename, so as written the notebook gives you no way to identify *which* radiograph blew up in Fold 3. Fixing this is the single most useful change (snippet in §7).

**Patient-level leakage is possible.** K-fold is done at the *image* level. Filenames such as `59_0.png` suggest a `patientID_index` scheme; if any patient contributed more than one image, two images of the same knee can land in different folds, leaking information. With 54 images this is a real risk and should be checked and, if present, replaced with grouped (patient-level) folds.

**Minor footguns.** The class defaults (`target_size=512`, `heatmap_scale=0.25`, `sigma=2.0`) differ from how it is always called (768 / 0.5 / 6.0), which will bite anyone who reuses it; hard-coded fallback dimensions (`width=2860`, `height=8000`) could silently misassign the hemisphere split if metadata is missing; and 6 × 2,000 = 12,000 epochs on 43 images is heavy — the good folds plateau by ~epoch 700–900, so early stopping would cut compute substantially with no loss of quality.

---

## 4. Research quality

The study design is the right shape for the question. Reporting the downstream surgical angle, not just landmark error, is the most important methodological strength — it is the quantity a surgeon acts on, and a sub-degree MAE is a genuinely strong result. PCK at multiple thresholds is appropriate for landmark work, and the one-stage "global" framing (regressing all 12 keypoints directly on the full radiograph, bypassing a YOLO sub-crop) is a reasonable contribution.

The weaknesses are about scale and evidence, not method. With 54 images and ~11-image folds, the cross-validation is more a stability probe than a precise generalisation estimate, and the variance it surfaces (Fold 3) is partly a small-sample artefact. There is no baseline in this notebook — in particular no comparison against the two-stage YOLO pipeline the abstract says this replaces — so the central claim is asserted rather than demonstrated. There is no per-keypoint error breakdown, no ground-truth reliability/inter-observer analysis, and no statistical treatment of uncertainty (with n=11 hemispheres per fold and in the test set, confidence intervals matter more than point estimates). The final 6-image test set is too small to carry a headline number without a stated confidence interval, and it is not held out from the CV folds, so it is only fully independent if no decisions were made on the basis of the CV results.

---

## 5. The performance outlier — analysis

### 5.1 The numbers

| Fold | n (hemispheres) | Best val MSE (px²) | PCK@0.01 | Angle MAE (°) | Angle Std (°) | Angle Max (°) |
|------|----------------:|-------------------:|---------:|--------------:|--------------:|--------------:|
| 1 | 21 | 6.63 | 99.2% | 0.581 | 0.383 | 1.302 |
| 2 | 21 | 4.60 | 100.0% | 0.377 | 0.303 | 1.087 |
| **3** | **21** | **32.28** | **98.5%** | **0.959** | **2.025** | **9.878** |
| 4 | 22 | 4.08 | 100.0% | 0.562 | 0.416 | 1.492 |
| 5 | 20 | 4.00 | 100.0% | 0.389 | 0.207 | 0.739 |
| **Reported mean ± std** | | 10.32 ± 11.02 | 99.5 ± 0.6% | 0.574 ± 0.210 | 0.667 ± 0.683 | 2.900 ± 3.498 |
| **Excluding Fold 3** | | ≈ 4.83 | 99.8% | 0.477 | 0.327 | 1.155 |
| Held-out test set | 11 (6 imgs) | — | 100.0% | 0.465 | 0.269 | 0.991 |

With Fold 3 removed, every fold sits in a tight band (MSE 4.0–6.6 px², angle MAE 0.38–0.58°, max 0.74–1.49°) that is fully consistent with the test-set result. Fold 3 is the lone outlier on every metric.

### 5.2 What kind of failure it is

The combination is the key. If Fold 3's model were uniformly worse, its PCK@0.01 would collapse and its MAE would be several degrees. Instead PCK@0.01 is 98.5% (≈1.5% of visible keypoints beyond the 7.7 px threshold — on the order of one or two keypoints across the whole fold) and the MAE is ~1°, while the **maximum** angular error is 9.878° and the std is 2.025°. That is a heavy right tail: almost all cases behave normally, and a single case (one mislocalised landmark) produces the entire excursion. The same single bad landmark, off by tens of pixels, also explains the MSE jump from ~5 to ~32 px² essentially by itself.

Crucially, 32.28 px² is the *minimum* MSE over all 2,000 epochs, and the late-epoch plateau hovers around 33–37 px², not 5. So the model never fit that case at any epoch — this is a **persistent** failure on a specific image, not a transient training spike. That points away from "bad luck in optimisation" and toward "a case the model genuinely cannot place," i.e. an atypical or mislabeled radiograph that is unlike the 43 training images in that particular split.

### 5.3 Why a small pixel error becomes a 10° error

The correction angle is computed as the angle between two vectors anchored at the osteotomy hinge point: `α = ∠(ankle_center − ost_point, target_at_ankle − ost_point)`, where `target_at_ankle` is itself the intersection of the femoral-head→Fujisawa line with the ankle horizontal. The angle is therefore a *lever*: a landmark near the hinge (the osteotomy point or the femoral head) has a short moment arm, so a localisation error of a few tens of pixels there rotates the vector by several degrees. Landmarks that the model usually nails to within ~2 px contribute negligibly, but a single miss on a leverage-sensitive landmark — exactly what a hard image produces — is amplified. This is intrinsic to the clinical measurement, not a coding bug, and it is why the *angular* outlier is larger in relative terms than the *pixel* outlier.

### 5.4 Root cause, ranked

1. **Most likely — a single hard/atypical or mislabeled validation image in Fold 3's split**, mislocalised on one leverage-sensitive landmark and amplified by the geometry. This is the explanation most consistent with the full signature (persistent ~33 px² floor, PCK essentially intact, MAE ~1°, max ~10°). Candidate causes for the image being hard: surgical hardware/implant, low contrast, unusual rotation or limb alignment, bilateral-vs-unilateral oddity, or a ground-truth labeling error.
2. **Enabling condition — the dataset is tiny.** With 11 validation cases per fold, one bad image moves the mean, std, and max dramatically; this is why the fold-to-fold spread is large and why the aggregate std exceeds its mean.
3. **Minor / cannot be excluded — optimisation variance.** Fresh per-fold initialisation plus non-reproducible, unseeded augmentation introduces run-to-run noise. Because a persistent MSE floor of ~33 is unlikely from optimisation noise alone (a noisy run would still reach ~5 px² at *some* epoch), this is a secondary contributor at most, but it cannot be ruled out until the pipeline is made reproducible.

---

## 6. How to report it in the paper

Do not hide or quietly drop the fold — reviewers will (rightly) ask, and a transparent failure analysis strengthens the paper.

**Report all five folds individually.** Show the per-fold table (§5.1), not just an aggregate. Let the reader see that four folds are tight and one is an outlier.

**Replace the misleading aggregate.** Report the central tendency as **median [min–max]** or report mean ± std *with an explicit statement that the standard deviation is inflated by Fold 3*. Provide a sensitivity figure showing the aggregate **with and without** Fold 3 — clearly labeled, and never use the Fold-3-excluded number as the headline.

**Lead with robust, clinically meaningful metrics.** Make the primary endpoints (a) the **proportion of cases within clinical tolerance** (e.g. % of hemispheres with |error| ≤ 1° and ≤ 2°), (b) **median and 95th-percentile** angular error, and (c) the **maximum** error with the count of cases above tolerance. Surgeons care about the worst case and the failure rate, not a px² mean. De-emphasise MSE-in-px² or move it to a supplementary table.

**Identify and characterise the offending case.** Use the per-image logging fix (§7) to find the specific radiograph, then include an overlay figure (predicted vs ground-truth landmarks) and state which landmark failed and why. If it is a labeling error, correct it and re-run, and say so. If it is a genuine hard case (implant, atypical anatomy), keep it and present it as a **named failure mode** — "the model's largest error (9.9°) arose from mislocalisation of the [landmark] on [image characteristic], reflecting both the scarcity of such cases in training and the leverage of that landmark in the α computation."

**State the statistics honestly.** With n=11 per fold (and 11 in the test set), give **bootstrap confidence intervals** rather than relying on mean ± std; the outlier will widen the CI, which is the correct and honest outcome.

**Frame sample size as a limitation.** Explicitly note that 54 images / ~11-image folds mean a single case dominates fold-level metrics, that this is the reason for the high fold variance, and that a larger external validation cohort is needed before clinical claims.

Suggested wording: *"Five-fold cross-validation gave a mean correction-angle MAE of 0.57° (median 0.56°, range 0.38–0.96° across folds). One fold contained a single outlier case with a 9.9° error, driven by mislocalisation of the [landmark]; this case also accounts for that fold's elevated keypoint MSE while PCK@0.01 remained 98.5%. Excluding this case, all folds fell within 0.38–0.58° MAE, consistent with the held-out test set (0.47°, max 0.99°). Given the cohort size (n=54), per-fold metrics are sensitive to individual cases; results require confirmation on a larger external dataset."*

---

## 7. Concrete fixes

**Capture per-image errors so the outlier can be found (highest value):**

```python
def evaluate_fold_angles(model, v_loader, dev):
    model.eval()
    per_case = []                       # NEW: keep every case, not just aggregates
    with torch.no_grad():
        for batch in v_loader:
            # ... existing inference ...
            for b in range(len(imgs)):
                for base in (0, 6):
                    # ... build pts_gt / pts_pred ...
                    if complete:
                        err = abs(pred_alpha - gt_alpha)
                        per_case.append({
                            "filename": os.path.basename(batch["img_path"][b]),
                            "side": "L" if base == 0 else "R",
                            "gt": gt_alpha, "pred": pred_alpha, "abs_err": err,
                        })
    if not per_case:
        return None
    errs = np.array([c["abs_err"] for c in per_case])
    worst = max(per_case, key=lambda c: c["abs_err"])
    return {"mae": errs.mean(), "std": errs.std(), "max": errs.max(),
            "n": len(errs), "per_case": per_case, "worst": worst}   # NEW
```

Then, after CV, print the worst case per fold and plot its overlay.

**Make training reproducible and seed folds independently:**

```python
def seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s); random.seed(s)

g = torch.Generator(); g.manual_seed(SEED)
DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
           worker_init_fn=seed_worker, generator=g, **_kw)

# inside the fold loop, before building the model:
set_seed(SEED + fold)
```

**Prevent patient-level leakage** (if a patient has >1 image):

```python
from sklearn.model_selection import GroupKFold
groups = [int(os.path.basename(s["img_path"]).split("_")[0]) for s in full_ds.samples]
for fold, (tr, va) in enumerate(GroupKFold(n_splits=5).split(np.arange(len(groups)), groups=groups)):
    ...
```

**Report robust metrics** alongside MSE — median/percentile Euclidean error in px, and the fraction of hemispheres within 1°/2°.

---

## 8. Bottom line

The model and the core method are good, and the held-out test result (0.47° MAE, 0.99° max) is clinically strong. Fold 3 is not evidence of a broken model; it is a single hard or mislabeled validation image whose one bad landmark is amplified by the angle geometry and magnified by a very small dataset. Report it transparently — all five folds, robust and clinically framed metrics, the identified failure case, confidence intervals, and an explicit sample-size limitation — and it becomes a credible failure-mode analysis rather than a weakness. The most valuable immediate change is logging per-image errors so the offending radiograph can be named and shown; the most valuable change for the paper's credibility is a larger, ideally external, validation cohort.
