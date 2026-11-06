# 提交台账（Submission Ledger）

> 每次线上提交务必记录：文件名、SHA-256、基线、线上总分、S_pos、S_p0、S_clean、回滚点。
> 线上 52.89 的实测基线是 Phase-1 文件；后续所有改动都应在该分数上做 A/B。

| 日期 | 文件 | SHA-256 (前 12 位) | 基线 / 改动 | 线上总分 | S_pos | S_p0 | S_clean | 备注 |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| 2026-11-06 | `results/final/pred_nriqa_dataset_test_phase1_breakthrough.json` | `6ecf6d4f74d4` | Phase-1 突破版 | **52.89** | 16.51 | 16.38 | 20.00 | 已实测，当前最高分 |
| 2026-11-06 | `results/final/pred_nriqa_dataset_test_qualirag_consensus.json` | `a3da92ed6ed7` | Phase-1 + 10 处 QualiRAG 黄金改动 | 待验证 | — | — | — | b983d87；15 Clean |
| 2026-11-06 | `results/final/pred_nriqa_dataset_test_FINAL_d1a437a.json` | `c72849e8e91b` | b983 + 8 处 ≥3/6 共识修订 | **52.70** | — | — | — | commit `9323007`；PDF 尸检确认比 52.89 低 0.19 |
| 2026-11-06 | `results/final/pred_nriqa_dataset_test_PHASEA.json` | `7c8bc46c87c7` | Phase-1 基线 + 28 处 Phase A 修订（16 Clean / P0 103 / P1 82） | 待验证 | — | — | — | 修复 Bug A/C/D + Sinkhorn；盲测 52.790（+0.832 vs Phase-1） |
| 2026-11-06 | `results/final/pred_nriqa_dataset_test_PHASEB.json` | `946919af3cca` | Phase-1 基线 + Phase B V12（原生物理+SA-AOR）+ 28 处修订（16 Clean / P0 102 / 345 positive） | 待验证 | — | — | — | Phase B 盲测 52.938（+0.980 vs Phase-1，CI [+0.700,+1.259]） |
| — | `results/final/pred_candidate_base_phase1.json` | — | Phase-1 + 11 处修订 | 备选 | — | — | — | 16 Clean；与 b983 不同基线，仅在需要时对照 |
| — | `results/final/pred_nriqa_dataset_test_55plus_unconstrained.json` | — | 全路由无约束 | **不要提交** | — | — | — | 会新增 P0、改动 P0 集合 |

## 回滚顺序

1. 若最终候选线上低于 52.89：回滚 `pred_nriqa_dataset_test.json` 到 `pred_nriqa_dataset_test_phase1_breakthrough.json`（52.89 实测）。
2. 若只想保留 QualiRAG 黄金改动：回滚到 `pred_nriqa_dataset_test_qualirag_consensus.json`（b983d87）。
3. 若需重新生成：见 `docs/D1A437A_FINAL_RESULT_REPORT.md` 第 8 节。
