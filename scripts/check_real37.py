"""在真实 37 例队列上验证置换检验（M2 验收）。

目标：复现已知结论——37 例 pilot 的名义 AUC ≈0.92，但在同流程标签置换下
p≈0.07，即**证据不足**。这一步验证的不是模型性能，而是工具能不能拦住虚报。

合规：真实谱与特征矩阵不进仓库，结果只写到 reports/（已在 .gitignore 中）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PIPE_SRC = Path(
    os.environ.get("MALDI_PIPELINE_SRC", str(ROOT.parent / "maldi-ms-pipeline" / "src"))
)
sys.path.insert(0, str(PIPE_SRC))

from maldi_ms.io import load_config  # noqa: E402

from maldi_ms_agent.session import AnalysisSession  # noqa: E402
from maldi_ms_agent.stats import permutation_test, summarize_null  # noqa: E402
from maldi_ms_agent.tools import build_tools  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    # 真实队列数据属未发表数据，路径不写进仓库，由命令行或环境变量提供
    ap.add_argument(
        "--data-dir",
        default=os.environ.get("MALDI_REAL_DATA_DIR", ""),
        required=not os.environ.get("MALDI_REAL_DATA_DIR"),
        help="真实谱数据目录（也可用环境变量 MALDI_REAL_DATA_DIR）",
    )
    ap.add_argument(
        "--config",
        default=str(ROOT.parent / "maldi-ms-pipeline" / "configs" / "uti8_50.yaml"),
    )
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "reports" / "real37_permutation.json"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    session = AnalysisSession(cfg=cfg)

    t0 = time.time()
    session.load(args.data_dir)
    print(session.summarize_dataset())

    session.build_features()
    print(session.summarize_features())
    print(f"[耗时] 读谱+峰检测+对齐+过滤 {time.time()-t0:.1f}s")

    tools = {t.name: t for t in build_tools(session)}
    r = tools["evaluate_panel"].func()
    print("\n--- evaluate_panel ---")
    print(r.text)

    t1 = time.time()
    rp = tools["permutation_test"].func(n_perm=args.n_perm, seed=args.seed)
    print(f"\n--- permutation_test（{time.time()-t1:.1f}s）---")
    print(rp.text)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "n_perm": rp.payload["n_perm"],
                "observed_auc": rp.payload["observed_auc"],
                "p_value": rp.payload["p_value"],
                "null_mean": rp.payload["null_mean"],
                "significant": rp.payload["significant"],
                "n_samples": int(session.frame.shape[0]),
                "n_features": int(session.frame.shape[1]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
