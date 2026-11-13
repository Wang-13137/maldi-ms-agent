"""评测与数据治理流水线（内建）。

把一次评测从"跑完拿到一个数字"变成三件**可执行**的事：

1. 这份结果用的是哪份数据 —— 内容哈希锚定的数据版本 + 运行血缘；
2. 差异是真的还是噪声 —— 按 case 配对做 McNemar 精确检验，不看两个比率谁大；
3. 能不能发 —— 通过率下限 / 不得显著回归 / 必须调用工具 三类门禁，未过退出码 1。

数据与评测的登记、校验由 ``maldi-ms-platform`` 提供（零第三方运行时依赖的内部库）。
本脚本是它在 agent 项目里的唯一入口：CI 跑它、本地跑它，两边结论必须一致。

两类数据在这里被区别对待：
- **评测报告**（合成扰动数据上跑出的 64 条）可以进仓库、可以下发；
- **真实受试者数据**只登记元数据（名称、批次数、样本数），payload 导出在库层被拒。

退出码：0 = 全部符合预期；1 = 有断言不成立（结论或门禁行为变了）；2 = 环境或产物校验失败。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "eval"
REPORT_PATH = EVAL_DIR / "released" / "agent_glm4flash_64_v1.json"
CORPUS_PATH = EVAL_DIR / "corpus" / "cases.json"
MANIFEST_PATH = EVAL_DIR / "released" / "manifest.json"
POLICY_PATH = EVAL_DIR / "policy.json"

PLATFORM_ENV = "MALDI_PLATFORM_SRC"


def _ensure_platform() -> None:
    """把平台加到 sys.path。

    优先环境变量 ``MALDI_PLATFORM_SRC``（CI 里指向 checkout 下来的兄弟仓库），
    其次默认同级目录 ``../maldi-ms-platform/src``。找不到就直接失败——
    宁可红，也不要静默跳过治理。
    """
    import os

    cands: list[Path] = []
    env = os.environ.get(PLATFORM_ENV, "").strip()
    if env:
        cands.append(Path(env))
    cands.append(ROOT.parent / "maldi-ms-platform" / "src")

    for cand in cands:
        if (cand / "maldi_platform" / "__init__.py").exists():
            sys.path.insert(0, str(cand))
            return
    raise RuntimeError(
        f"找不到 maldi-ms-platform：检查环境变量 {PLATFORM_ENV}，"
        f"或把仓库放在 {ROOT.parent / 'maldi-ms-platform'}"
    )


def _sha256(path: Path) -> str:
    """按 LF 计算文件哈希。

    跨平台内容寻址的坑：同一份 JSON 在 Windows 上是 CRLF、在 Linux 上被 git 存成 LF，
    按原始字节算会得到两个身份，于是"本地通过、CI 红"。这里统一归一化换行符，
    再配合 `.gitattributes` 把这两个目录锁成 eol=lf。
    """
    import hashlib

    h = hashlib.sha256()
    tail = b""
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            data = tail + chunk
            # \r 恰好落在块边界时留到下一轮，避免把一个 CRLF 拆成两半漏掉
            if data.endswith(b"\r"):
                tail, data = b"\r", data[:-1]
            else:
                tail = b""
            h.update(data.replace(b"\r\n", b"\n"))
    if tail:
        h.update(tail)
    return h.hexdigest()


def verify_artifacts() -> list[dict[str, Any]]:
    """按 manifest 校验冻结产物的哈希。

    报告是结论的唯一来源，被改过一个字节就不能再拿它的数字说话。
    """
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    out = []
    for item in manifest["artifacts"]:
        path = ROOT / item["path"]
        if not path.exists():
            raise RuntimeError(f"冻结产物缺失：{item['path']}")
        actual = _sha256(path)
        out.append(
            {
                "path": item["path"],
                "expected": item["sha256"],
                "actual": actual,
                "ok": actual == item["sha256"],
            }
        )
    return out


class Checker:
    """收集断言结果：全部记录，最后统一决定退出码，避免第一个失败就丢掉后续证据。"""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def expect(self, name: str, actual: Any, expected: Any) -> bool:
        ok = actual == expected
        self.items.append({"name": name, "expected": expected, "actual": actual, "ok": ok})
        mark = "通过" if ok else "未通过"
        print(f"  [{mark}] {name}: expected={expected!r} actual={actual!r}")
        return ok

    @property
    def ok(self) -> bool:
        return all(i["ok"] for i in self.items)


def run_governance(workdir: Path, out_path: Path, policy: dict[str, Any]) -> dict[str, Any]:
    from maldi_platform import audit, datasets, db, evals

    t0 = time.perf_counter()
    db_path = workdir / "governance.db"
    if db_path.exists():
        db_path.unlink()
    conn = db.init_db(db_path)
    schema_version = db.current_version(conn)

    # ---- 1. 数据中台：语料登记（合成数据，可开放导出）------------------------
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    ds_synth = datasets.register_dataset(
        conn,
        "agent-perturbation-corpus",
        classification="synthetic",
        export_policy="open",
        modality="synthetic-maldi",
        description="扰动评测语料：4 类任务 × 4 seed，全部由 synth.py 确定性生成",
    )
    ver_synth = datasets.add_version(
        conn,
        ds_synth,
        "v1",
        source=EVAL_DIR / "corpus",
        n_samples=corpus["n_cases"],
        notes=f"content address; {corpus['n_cases']} cases",
    )
    audit.record(conn, "dataset.add", target="agent-perturbation-corpus@v1", detail="synthetic/open")

    for name, role in (("case_id", "id"), ("task_class", "meta"), ("expected", "label")):
        datasets.add_field(conn, ver_synth, name, role=role)

    # ---- 2. 数据中台：真实数据只登记元数据 ----------------------------------
    ds_real = datasets.register_dataset(
        conn,
        "uti-urine-real37",
        classification="real",
        export_policy="internal",
        modality="MALDI-TOF",
        description="37 例尿液样本（深圳大学总医院采集，rapifleX）。原始谱与特征矩阵不入库",
    )
    ver_real = datasets.add_version(
        conn,
        ds_real,
        "v1",
        n_samples=37,
        acquisition_batch="B1",
        notes="仅元数据；payload 不得出域",
    )
    audit.record(conn, "dataset.add", target="uti-urine-real37@v1", detail="real/internal")

    # ---- 3. 评测中台：套件绑数据版本 + 报告入库 ----------------------------
    suite = evals.create_suite(
        conn,
        policy["suite"],
        dataset_version_id=ver_synth,
        task_type="agent-judgement",
        metric_spec={"primary": "pass_rate", "paired": "mcnemar_exact"},
    )
    run_ids = evals.ingest_file(conn, suite, str(REPORT_PATH))
    runs = {evals.get_run(conn, rid)["arm"]: evals.get_run(conn, rid) for rid in run_ids}
    audit.record(conn, "run.ingest", target=REPORT_PATH.name, detail=f"suite={suite}; runs={run_ids}")

    base_arm = policy["baseline_arm"]
    cand_arm = policy["candidate_arm"]
    if base_arm not in runs or cand_arm not in runs:
        raise RuntimeError(f"报告里找不到臂 {base_arm} / {cand_arm}，实际有 {sorted(runs)}")
    base_id, cand_id = runs[base_arm]["id"], runs[cand_arm]["id"]

    # ---- 4. 配对比较（不是比两个比率大小）----------------------------------
    cmp = evals.compare_runs(conn, base_id, cand_id)  # a=基线, b=候选
    table = cmp["table"]

    # ---- 5. 三类门禁 -------------------------------------------------------
    g_min = evals.gate_min_pass_rate(conn, base_id, policy["min_pass_rate"])
    tool = policy["require_tool"]
    g_tool_base = evals.gate_require_tool(conn, base_id, tool["metric"], tool["threshold"])
    g_tool_cand = evals.gate_require_tool(conn, cand_id, tool["metric"], tool["threshold"])
    g_reg_cand = evals.gate_no_regression(
        conn, cand_id, base_id, alpha=policy["alpha"], name="no_regression"
    )
    g_reg_base = evals.gate_no_regression(
        conn, base_id, cand_id, alpha=policy["alpha"], name="no_regression"
    )
    for gate in (g_min, g_tool_base, g_tool_cand, g_reg_cand, g_reg_base):
        audit.record(
            conn,
            "gate.evaluate",
            target=gate["name"],
            outcome="ok" if gate["passed"] else "denied",
            detail=gate["detail"][:200],
        )

    # ---- 6. 合规：真实数据 payload 一律拒 ----------------------------------
    denied_msg = ""
    try:
        datasets.check_export(conn, ver_real, "payload")
        export_outcome = "allowed"
    except datasets.ExportDeniedError as exc:
        export_outcome = "denied"
        denied_msg = str(exc)
    audit.record(conn, "dataset.export", target="uti-urine-real37@v1", outcome="denied",
                 detail="payload 导出被拒（真实数据只登记元数据）")

    # 审计只追加：库层触发器必须挡住改删
    append_only = True
    for stmt in ("UPDATE audit_log SET outcome='ok'", "DELETE FROM audit_log"):
        try:
            conn.execute(stmt)
            conn.commit()
            append_only = False
        except sqlite3.DatabaseError:
            conn.rollback()
    audit_rows = audit.list_entries(conn, limit=100)

    # ---- 7. 汇总 -----------------------------------------------------------
    report = {
        "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "platform": {"schema_version": schema_version, "db": str(db_path.name)},
        "dataset": {
            "corpus": {
                "name": "agent-perturbation-corpus@v1",
                "version_id": ver_synth,
                "content_hash": datasets.describe_version(conn, ver_synth)["version"]["content_hash"],
                "n_cases": corpus["n_cases"],
                "classification": "synthetic",
                "export_policy": "open",
            },
            "real": {
                "name": "uti-urine-real37@v1",
                "version_id": ver_real,
                "content_hash": datasets.describe_version(conn, ver_real)["version"]["content_hash"],
                "n_samples": 37,
                "classification": "real",
                "export_policy": "internal",
                "payload_export": export_outcome,
                "denied_reason": denied_msg,
            },
        },
        "suite": {"id": suite, "name": policy["suite"], "task_type": "agent-judgement"},
        "runs": [
            {
                "arm": arm,
                "run_id": r["id"],
                "label": r["label"],
                "n": r["n_cases"],
                "k": r["n_passed"],
                "rate": round(r["n_passed"] / r["n_cases"], 4) if r["n_cases"] else 0.0,
            }
            for arm, r in sorted(runs.items())
        ],
        "compare": {
            "baseline": cmp["a"],
            "candidate": cmp["b"],
            "table": table,
            "mcnemar_p": cmp["mcnemar_p"],
            "verdict": cmp["verdict"],
            "by_task_class": cmp["by_task_class"],
        },
        "gates": {
            "min_pass_rate_baseline": g_min,
            "require_tool_baseline": g_tool_base,
            "require_tool_candidate": g_tool_cand,
            "no_regression_candidate_vs_baseline": g_reg_cand,
            "no_regression_baseline_vs_candidate": g_reg_base,
        },
        "audit": {
            "n_entries": len(audit_rows),
            "denied": sum(1 for r in audit_rows if r["outcome"] == "denied"),
            "append_only": append_only,
        },
    }

    chk = Checker()
    print("\n断言：")
    chk.expect("候选相对基线显著变差", table["a_only"] > table["b_only"] and cmp["mcnemar_p"] < policy["alpha"], True)
    chk.expect("候选门禁：no_regression 拦住", g_reg_cand["passed"], False)
    chk.expect("反向门禁：基线相对候选放行", g_reg_base["passed"], True)
    chk.expect("候选门禁：必须调用置换检验", g_tool_cand["passed"], False)
    chk.expect("基线门禁：必须调用置换检验", g_tool_base["passed"], True)
    chk.expect("基线门禁：通过率下限", g_min["passed"], True)
    chk.expect("真实数据 payload 导出", export_outcome, "denied")
    chk.expect("审计日志只追加", append_only, True)
    chk.expect("审计留痕不少于 4 类动作", len(audit_rows) >= 4, True)

    report["checks"] = chk.items
    report["ok"] = chk.ok
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n臂级结果：")
    for row in report["runs"]:
        print(f"  {row['arm']:<16} {row['k']}/{row['n']}  rate={row['rate']:.4f}")
    print(
        f"\n配对比较 {cmp['a']['label']} vs {cmp['b']['label']}："
        f"仅基线过={table['a_only']} 仅候选过={table['b_only']} p={cmp['mcnemar_p']}"
    )
    print(f"审计：{len(audit_rows)} 条，denied={report['audit']['denied']}，只追加={append_only}")
    print(f"报告：{out_path}")
    print(f"结论：{'全部符合预期' if chk.ok else '存在不符合预期的断言'}")
    conn.close()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="评测与数据治理流水线（内建）")
    parser.add_argument("--out", default=str(ROOT / "reports" / "governance.json"))
    parser.add_argument("--workdir", default=None, help="工作目录（默认临时目录，跑完即删）")
    parser.add_argument("--keep-db", action="store_true", help="保留下游数据库，便于人工检查")
    args = parser.parse_args(argv)

    try:
        _ensure_platform()
    except RuntimeError as exc:
        print(f"环境不满足：{exc}", file=sys.stderr)
        return 2

    try:
        artifacts = verify_artifacts()
    except RuntimeError as exc:
        print(f"产物校验失败：{exc}", file=sys.stderr)
        return 2
    bad = [a for a in artifacts if not a["ok"]]
    if bad:
        for a in bad:
            print(
                f"产物哈希不符：{a['path']}\n  manifest={a['expected']}\n  actual  ={a['actual']}",
                file=sys.stderr,
            )
        return 2

    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    tmp: tempfile.TemporaryDirectory[str] | None = None
    if args.workdir:
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.TemporaryDirectory(prefix="maldi-governance-")
        workdir = Path(tmp.name)

    try:
        report = run_governance(workdir, Path(args.out), policy)
    finally:
        if tmp is not None and not args.keep_db:
            try:
                tmp.cleanup()
            except OSError as exc:  # Windows 上句柄释放有延迟，清理失败不该让门禁变红
                print(f"临时目录未能清理（不影响结论）：{exc}", file=sys.stderr)
        elif tmp is not None:
            print(f"工作目录保留：{workdir}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
