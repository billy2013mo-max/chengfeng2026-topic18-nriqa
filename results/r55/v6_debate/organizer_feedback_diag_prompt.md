# Diagnostic & Improvement Alignment: Organizer Feedback on `resident_v6` (644 Final Eval)

## 1. Official Organizer Feedback on Our First 644-Image Submission
We submitted `pred_nriqa_dataset_final_644_r55_resident_v6_routed_gate_eu.json` (fresh 8-GPU inference of `resident_v6` 9-model ensemble + `RoutedGateEUDecoder`).
Organizers' feedback:
- **Objective Total**: `53.99 / 65` (passed initial baseline)
- **S_pos (14-class Macro-F1 × 25)**: `17.16 / 25` (implies mean 14-class F1 = `0.6864`)
- **S_p0 (6-class P0 F1 × 20)**: `16.83 / 20` (implies mean 6-class P0 F1 = `0.8415`, and mean 8-class P1/P2 F1 ≈ `0.570`)
- **Clean Score**: `20.00 / 20`
- **Critical Warning**: *"需优化p1、p2的表现，以及过度生成、虚化突兀、过锐表现"* — specifically **过度生成 (overgen), 虚化突兀 (bokeh), 过锐 (oversharpening)** are currently too low and risk failing an upcoming per-dimension baseline adjustment!

Constraints:
- `AGENTS.md` Rules 1–3 strictly apply: ZERO tuning/calibration on 644-set or test_150 distributions; ZERO legacy JSON anchoring; all changes must be validated strictly on 8-fold OOF / tag-split OOF.
- Deployment: single RTX 4090 (`<= 24 GiB` VRAM, `<= 15s/img`, `<= 30B` active params). We currently have 8×H200 GPUs idle and ready for targeted training/refinement.

---

## 2. Exact 8-Fold OOF Diagnostic of `resident_v6` (`LAMS_V6`)

### A. Per-Class F1 on `ORIG (N=1902)` & `ALL (N=3573)`
`ORIG`: `Doc2_Strict=52.034 | S_pos=15.848/25 | S_p0=16.380/20 | Clean2=19.805/20`
- **模糊**: `P0=0.904 (P=0.903, R=0.906)`, `P1=0.560 (P=0.608, R=0.519)`, `P2=0.433 (P=0.369, R=0.525)`
- **纹理损失**: `P0=0.835 (P=0.816, R=0.854)`, `P1=0.607 (P=0.593, R=0.622)`, `P2=0.628 (P=0.609, R=0.648)`
- **过度生成**: `P0=0.757 (P=0.702, R=0.821)`, `P1=0.458 (P=0.434, R=0.485)`, `P2=0.516 (P=0.454, R=0.599)`
- **虚化突兀**: `P0=0.977 (P=0.991, R=0.963)` (`supp=109, pred=106, TP=105, FP=1, FN=4`)
- **过锐**: `P0=0.563 (P=0.583, R=0.544)`, `P1=0.363 (P=0.382, R=0.345)`, `P2=0.438 (P=0.397, R=0.490)`
- **花屏**: `P0=0.835 (P=0.771, R=0.910)`
- Summary: `Mean P0 F1 = 0.8118`, `Mean P1 F1 = 0.4970`, `Mean P2 F1 = 0.5040`.

On `supp` split (`N=1671`), `过锐` collapses further: `P0=0.614 (pred=180 vs supp=100)`, `P1=0.240 (R=0.210)`, `P2=0.179 (R=0.157)`.
On `tag_fold` (unseen tag split, `ORIG`), `虚化突兀` is STILL `P0=0.982 (supp=109, pred=109, P=0.982, R=0.982)`.

### B. Confusion Matrices on `ORIG (1902)` (rows=True `[无,P2,P1,P0]`, cols=Pred `[无,P2,P1,P0]`)
- **过度生成**:
  - `T_无`: `1167 无, 147 P2, 25 P1, 3 P0` (147 false-alarm P2s!)
  - `T_P0`: `0 无, 5 P2, 15 P1, 92 P0`
  - `T_P1`: `6 无, 49 P2, 82 P1, 32 P0`
  - `T_P2`: `41 无, 167 P2, 67 P1, 4 P0`
- **过锐**:
  - `T_无`: `1468 无, 65 P2, 9 P1, 8 P0`
  - `T_P0`: `4 无, 16 P2, 21 P1, 49 P0`
  - `T_P1`: `19 无, 30 P2, 39 P1, 25 P0`
  - `T_P2`: `41 无, 73 P2, 33 P1, 2 P0`

### C. `lam_cl` Sensitivity Sweep on `Q_oof` (ORIG)
- `argmax`: `Doc2=51.002 | S_pos=15.56 | S_p0=15.99 | Cln=19.45 | 过生=0.55/0.47/0.75 | 过锐=0.34/0.33/0.51`
- `lam=0.50`: `Doc2=51.571 | S_pos=15.90 | S_p0=16.37 | Cln=19.30`
- `lam=1.25`: `Doc2=51.764 | S_pos=15.86 | S_p0=16.33 | Cln=19.57`
- `lam=1.75`: `Doc2=51.975 | S_pos=15.89 | S_p0=16.40 | Cln=19.69`
- `lam=3.00`: `Doc2=52.033 | S_pos=15.82 | S_p0=16.37 | Cln=19.84`
So `lam_cl` is **NOT** what suppresses `P1/P2`, `过度生成`, or `过锐`!

### D. Per-Member 14-Class AP (`P0/P1/P2`) and Any-Defect (`>=1`) AP on `ORIG`
- `Q_oof_pool` Any-Defect AP: `模糊=0.936, 纹损=0.970, 过生=0.911, 虚化=0.997, 过锐=0.869, 花屏=0.910`
- `Q_oof_pool` 14-class AP (`P0/P1/P2`):
  - `模糊`: `0.96/0.60/0.44`
  - `纹损`: `0.93/0.64/0.62`
  - `过生`: `0.83/0.47/0.53` (best single members: `p1_qy` `0.81/0.44/0.50`, `p12_pe` `0.79/0.44/0.49`)
  - `虚化`: `1.00`
  - `过锐`: `0.61/0.36/0.41` (best single members: `ntg_pe` `0.55/0.35/0.37`, `p12_pe` `0.54/0.34/0.38`, `cradio_v3` `0.61/0.33/0.33`; weakest: `p3_qy` `0.48/0.31/0.33`, `p1_qy` `0.53/0.31/0.30`)
  - `花屏`: `0.91`

---

## 3. Questions for `claude` (`fable 5.2`)
Please give a rigorous, non-handwaving engineering & modeling diagnosis and an actionable plan:

1. **Why is `虚化突兀` `0.982` F1 on OOF (even `tag_fold`) yet flagged by the organizers as "too low" on the 644-image Final Test Set?**
   - On 644, `resident_v6` predicted `虚化突兀 P0 = 51 / 644 = 7.9%` (vs `5.7%` in `ORIG`).
   - What is the most likely failure mode on unseen 644 images (e.g., shallow-DoF natural photography / portrait bokeh FP vs new AIGC/ISP bokeh FN, or `Clean=20/20` masking non-clean FPs, or dataset artifact in `qingyu_8fold`)? How can we test/fix `虚化突兀` generalization without touching 644 labels? (Note we have `dual_engine_meta_aux_bkseg1.npz` and unlabelled/external images.)

2. **Why is `过锐` (Oversharpening) so bad across all levels (`P0=0.563, P1=0.363, P2=0.438`), and why does `过度生成` suffer on `P1/P2` (`0.458 / 0.516`)?**
   - For `过锐`: notice `p1_qy` and `p3_qy` are the *weakest* on `过锐` (tile/downsampled VLMs/ViTs destroy 1:1 edge-ringing & haloing!), while `ntg_pe` / `p12_pe` / `cradio_v3` are slightly better. Is this a **1:1 native-resolution / high-pass / edge-halo representation bottleneck**, a **label noise / ordinal boundary bottleneck**, or **loss weighting** (e.g., `TrueGateConstrainedEU` optimizes `Doc2_Strict` where `过锐` P0 weight = `20/6` = `3.33` vs `过锐` P1/P2 weight = `25/14` = `1.79`, or `p0_sample_boost` skews P0)?
   - For `过度生成`: Any-Defect AP is `0.911`, yet `P1=0.458`, `P2=0.516`, and `147` clean/无 images are false-alarmed as `P2`. Is this **head/loss** (`Q_oof` has good ranking, bad `无/P2/P1` boundary), or **feature representation** (e.g., `AIMv2` / `WebSSL` / `Qwen3-VL-32B` / `InternVL3_5-38B` / `ogsr1` aux)?

3. **Concrete Action Plan (Ranked by Expected Gain / Risk / Compute)**:
   - **Track A (Decoder / Calibration / Zero-Retrain or Fast-Retrain)**: Can per-dimension `TrueGateConstrainedEU` (e.g., per-defect `w_supp` or per-class threshold/bias / ordinal expectation decoding / `P1/P2` boundary calibration) improve `P1/P2`, `过锐`, and `过度生成` on 8-fold nested OOF without hurting `Clean2`? Or is the bottleneck purely upstream in `Q_oof`?
   - **Track B (Targeted High-Signal Representation / Specialist Training on 8×H200)**:
     - What specific specialist / feature / aux-data fix should we build for:
       (i) **`过锐` (1:1 native pixel crops + high-pass / Laplacian / Sobel / DINO-or-ConvNeXt-native residual branch + `ogsr1` aux?)**
       (ii) **`过度生成` (semantic + local plausibility: e.g., `32B/38B` VLM or `AIMv2` / `ogsr1` / `p1_qy` targeted head/loss for `无/P2/P1`?)**
       (iii) **`虚化突兀` (OOD robustness: hard-negative natural shallow-DoF bokeh + synthetic/segmentation-guided `bkseg1` boundary consistency?)**
       (iv) **`P1/P2` overall (loss function: currently `p0_sample_boost` + `beta_emd` — is `P1/P2` under-weighted during training?)**

Be concise, quantitative, and explicit about what to verify first on 8-fold OOF before launching 8×H200 runs.
