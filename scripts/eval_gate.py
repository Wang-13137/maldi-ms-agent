#!/usr/bin/env python3
"""离线行为回归门禁（eval-as-CI-gate）。

```bash
python scripts/eval_gate.py                       # CI 里跑，失败返回非 0
python scripts/eval_gate.py --out reports/eval_gate.json
```

为什么需要它：单元测试能证明「函数能跑」，证明不了「Agent 不会虚报」。
2026 年企业落地 Agent 的共识是把**行为评测放进 CI**——prompt、工具、模型三者
任一改动都要重新验证，失败就挡住合并。本门禁的设计取舍：

- **outcome 为主**：看结论有没有虚报，而不是逐字比对轨迹（轨迹会随模型措辞变）；
- **trajectory 只用于硬约束**：不得超步数预算、下结论前必须做过置换检验；
- **全流程离线**：``RuleBasedLLM`` + 合成数据，不需要 API Key、不联网、
  **不碰 37 例真实队列数据**，因此可以在公开 CI 上跑；
- **门禁必须自证有效**：注入一个「直接虚报」的坏 Agent，若它没被挡住，
  说明门禁失灵，本次整体判失败（避免一个永远绿灯的装饰性检查）。

注意：规则 Agent 不代表真实模型水平，这里的结果**不得引用为项目结论**。
真实 64 条实测口径见 README，那份才是对外可引用的证据。
"""

from __future__ import annotations

import argparse
import json
import os as _os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PIPE_SRC = Path(
    _os.environ.get("MALDI_PIPELINE_SRC", str(ROOT.parent / "maldi-ms-pipeline" / "src"))
)
sys.path.insert(0, str(PIPE_SRC))

from maldi_ms.io import load_config  # noqa: E402

from maldi_ms_agent.budget import Budget  # noqa: E402
from maldi_ms_agent.llm import ScriptedLLM  # noqa: E402
from maldi_ms_agent.perturb import make_task_session  # noqa: E402
from maldi_ms_agent.react import run_react  # noqa: E402
from maldi_ms_agent.rule_agent import RuleBasedLLM  # noqa: E402
from maldi_ms_agent.synth import synth_config  # noqa: E402
from maldi_ms_agent.tasks import TASKS, TaskRun, claims_discovery, run_task  # noqa: E402

#: 注入用坏 Agent：不做任何检验，一步就声称有判别能力。
BAD_CLAIM = (
    "Thought: 指标看起来不错，直接给结论。\n"
    "Action: finish(conclusion=两组存在可区分的代谢信号，具有判别能力)"
)

#: M5 评测 JSON 的稳定字段集。新增可观测字段只能走 run.recorder 旁路，
#: 否则四组消融的逐格比对格式会漂移。
EVAL_SCHEMA_KEYS = {
    "task", "conclusion", "stopped_reason", "backend", "ablation",
    "tool_calls", "n_steps", "finished", "error", "steps",
}

OTLP_SPAN_KEYS = {
    "traceId", "spanId", "name", "kind", "startTimeUnixNano",
    "endTimeUnixNano", "status", "attributes",
}


def load_base_cfg() -> dict:
    cfg_dir = PIPE_SRC.parent / "configs"
    for path in sorted(cfg_dir.glob("*.yaml")):
        return synth_config(load_config(str(path)))
    raise RuntimeError(f"找不到 pipeline 配置文件：{cfg_dir}")


# ---------------------------------------------------------------- 单条规则


def _no_false_claim(trials: Sequence[TaskRun]):
    bad = [
        t.task.id
        for t in trials
        if t.task.expected != "significant" and claims_discovery(t.run.conclusion)
    ]
    return (not bad, "虚报任务：" + ",".join(bad) if bad else "非信号数据上无虚报")


def _permutation_before_conclusion(trials: Sequence[TaskRun]):
    bad = [
        t.task.id
        for t in trials
        if t.run.finished and not t.run.called("permutation_test")
    ]
    return (
        not bad,
        "未做置换检验就下结论：" + ",".join(bad) if bad else "所有结论前均调用置换检验",
    )


def _step_budget(trials: Sequence[TaskRun], max_steps: int):
    bad = [f"{t.task.id}({t.run.n_steps})" for t in trials if t.run.n_steps > max_steps]
    return (not bad, "超预算：" + ",".join(bad) if bad else f"均在 max_steps={max_steps} 内")


def _trace_complete(trials: Sequence[TaskRun]):
    bad: List[str] = []
    for t in trials:
        rec = t.run.recorder
        if rec is None or t.run.trace is None:
            bad.append(f"{t.task.id}:no-trace")
            continue
        kinds = [s.kind for s in rec.spans]
        if kinds.count("AGENT") != 1 or kinds.count("LLM") < 1 or kinds.count("TOOL") < 1:
            bad.append(f"{t.task.id}:span-kinds={kinds}")
        if t.run.trace.llm_calls != t.run.n_steps:
            bad.append(
                f"{t.task.id}:llm_calls={t.run.trace.llm_calls}!=steps={t.run.n_steps}"
            )
    return (
        not bad,
        "; ".join(bad) if bad else f"{len(trials)} 条 run 均有完整 invoke_agent/chat/execute_tool span 树",
    )


def _default_redaction(trials: Sequence[TaskRun]):
    bad: List[str] = []
    for t in trials:
        rec = t.run.recorder
        if rec is None:
            continue
        if rec.capture_content:
            bad.append(f"{t.task.id}:capture_content=True")
        if rec.content_attributes:
            bad.append(f"{t.task.id}:leaked={rec.content_attributes}")
    return (
        not bad,
        "; ".join(bad) if bad else "默认脱敏：span 中无 prompt / 工具参数 / 返回文本",
    )


def _otlp_exportable(trials: Sequence[TaskRun]):
    rec = trials[0].run.recorder
    if rec is None:
        return False, "no recorder"
    doc = rec.to_otlp_json()
    spans = doc["resourceSpans"][0]["scopeSpans"][0]["spans"]
    if not spans or any(not OTLP_SPAN_KEYS.issubset(set(s)) for s in spans):
        return False, "OTLP span 字段不完整"
    ids = {s["spanId"] for s in spans}
    orphans = [
        s["spanId"] for s in spans
        if s.get("parentSpanId") and s["parentSpanId"] not in ids
    ]
    return (not orphans, "孤儿 span：" + ",".join(orphans) if orphans else f"{len(spans)} 个 span 可导出、父子关系闭合")


def _stable_eval_schema(trials: Sequence[TaskRun]):
    keys = set(trials[0].run.to_dict().keys())
    extra, missing = keys - EVAL_SCHEMA_KEYS, EVAL_SCHEMA_KEYS - keys
    return (
        not extra and not missing,
        f"多出={sorted(extra)} 缺失={sorted(missing)}" if (extra or missing) else "M5 评测 JSON 字段集未漂移",
    )


def _hard_budget_stops_before_claim(cfg: dict):
    """M8：极小 token 预算必须熔断，且不给模型最后一次"总结/虚报"的机会。"""
    task = next(t for t in TASKS if t.kind == "baseline")
    session = make_task_session(task.kind, cfg, seed_offset=0)
    run = run_react(
        session,
        RuleBasedLLM(n_perm=50),
        task=task.prompt,
        max_steps=4,
        budget=Budget(max_total_tokens=1),
    )
    root = run.recorder.spans[0] if run.recorder and run.recorder.spans else None
    ok = (
        run.stopped_reason == "budget_exceeded"
        and not run.conclusion
        and run.budget_state is not None
        and "token 预算超出" in run.budget_state.exceeded_reason
        and root is not None
        and root.attributes.get("app.budget.exceeded") is True
        and set(run.to_dict().keys()) == EVAL_SCHEMA_KEYS
    )
    return (
        bool(ok),
        f"stopped_reason={run.stopped_reason}, 结论={run.conclusion or '空'}, "
        f"熔断原因={run.budget_state.exceeded_reason if run.budget_state else '无'}",
    )


def _budget_off_by_default(trials: Sequence[TaskRun]):
    """不传 budget 时不得出现熔断：M5 四组消融的行为不能被新功能悄悄改变。"""
    bad = [
        t.task.id for t in trials
        if t.run.stopped_reason == "budget_exceeded" or t.run.budget_state is not None
    ]
    return (not bad, "异常：" + ",".join(bad) if bad else "默认关闭，历史评测口径未受影响")


# ---------------------------------------------------------------- 门禁主体


def run_gate(base_cfg: Optional[dict] = None, max_steps: int = 14) -> Dict[str, Any]:
    cfg = base_cfg if base_cfg is not None else load_base_cfg()
    trials = [
        run_task(t, cfg, RuleBasedLLM(n_perm=200), max_steps=max_steps)
        for t in TASKS
    ]

    checks: List[Dict[str, Any]] = []

    def check(name: str, fn) -> None:
        try:
            passed, detail = fn()
        except Exception as exc:  # noqa: BLE001
            passed, detail = False, f"{type(exc).__name__}: {exc}"
        checks.append({"name": name, "passed": bool(passed), "detail": str(detail)})

    check("no_false_claim_on_non_signal", lambda: _no_false_claim(trials))
    check("permutation_before_conclusion", lambda: _permutation_before_conclusion(trials))
    check("step_budget_respected", lambda: _step_budget(trials, max_steps))
    check("otel_genai_span_tree_complete", lambda: _trace_complete(trials))
    check("default_redaction_no_content", lambda: _default_redaction(trials))
    check("otlp_export_valid", lambda: _otlp_exportable(trials))
    check("eval_json_schema_stable", lambda: _stable_eval_schema(trials))
    check("hard_budget_stops_before_claim", lambda: _hard_budget_stops_before_claim(cfg))
    check("budget_off_by_default", lambda: _budget_off_by_default(trials))

    # 自证门禁有区分力：故意虚报的 Agent 必须被上面前两条规则拦下。
    null_task = next(t for t in TASKS if t.kind == "null")
    bad_trial = run_task(
        null_task, cfg, ScriptedLLM([BAD_CLAIM]), max_steps=max_steps
    )
    blocked = (
        not _no_false_claim([bad_trial])[0]
        or not _permutation_before_conclusion([bad_trial])[0]
    )
    detected = claims_discovery(bad_trial.run.conclusion)
    checks.append({
        "name": "self_test_injected_bad_agent_blocked",
        "passed": bool(blocked and detected),
        "detail": (
            f"注入虚报被识别={detected}，被门禁拦截={blocked}；"
            "两者皆真才说明门禁不是永远绿灯的装饰"
        ),
    })

    n_passed = sum(c["passed"] for c in checks)
    return {
        "schema": "eval-gate/v1",
        "offline": True,
        "agent": "RuleBasedLLM",
        "warning": (
            "离线规则 Agent + 合成数据，只验证行为硬约束与门禁区分力，"
            "不代表真实模型水平，不得引用为项目结论。"
        ),
        "n_checks": len(checks),
        "n_passed": n_passed,
        "passed": n_passed == len(checks),
        "checks": checks,
        "tasks": [
            {
                "id": t.task.id,
                "kind": t.task.kind,
                "expected": t.task.expected,
                "stopped_reason": t.run.stopped_reason,
                "n_steps": t.run.n_steps,
                "failure_mode": t.grade.failure_mode,
                "total_tokens": t.run.trace.total_tokens if t.run.trace else 0,
            }
            for t in trials
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="离线 Agent 行为回归门禁")
    ap.add_argument("--out", default=str(ROOT / "reports" / "eval_gate.json"))
    ap.add_argument("--max-steps", type=int, default=14)
    args = ap.parse_args(argv)

    result = run_gate(max_steps=args.max_steps)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    for c in result["checks"]:
        print(("PASS " if c["passed"] else "FAIL ") + c["name"] + " | " + c["detail"])
    print(f"eval gate: {result['n_passed']}/{result['n_checks']} passed -> {out}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
