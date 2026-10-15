"""手写 ReAct 循环。

为什么不直接用 LangChain / LangGraph：这个项目要评测的是**工具契约**与**提示词口径**
各自的贡献，中间层越少越容易归因；而且 ReAct 循环本身只有几十行，自己写可以完全
控制轨迹格式（M5 的四组消融都要逐格比对轨迹）。

循环结构：
    Thought → Action → Observation → Thought → ... → finish

停止条件：调用 finish / 达到 max_steps / 连续解析失败 / 工具级异常不可恢复。
不论以哪种方式停，轨迹都完整保留——**失败轨迹正是评测的对象**。
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .llm import LLM, Message, ScriptedLLM
from .prompts import build_system_prompt, build_user_prompt
from .session import AnalysisSession
from .tools import Tool, ToolResult, build_tools

# ---------------------------------------------------------------- 解析


#: 匹配 "Action: tool(a=1, b=x)"；允许 Action 前后有空格与 Markdown 加粗
_ACTION_RE = re.compile(
    r"^\s*\*{0,2}Action\*{0,2}\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE
)
_THOUGHT_RE = re.compile(
    r"^\s*\*{0,2}Thought\*{0,2}\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE
)
#: 兜底：模型写成 "Final Answer: ..."
_FINAL_RE = re.compile(
    r"^\s*\*{0,2}(?:Final\s*Answer|Answer|Conclusion)\*{0,2}\s*:\s*(.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)


class ParseError(ValueError):
    """模型输出解析失败。会被回喂给模型，让它自己纠正格式。"""


def _split_top_level(s: str) -> List[str]:
    """按顶层逗号切分参数串，引号内的逗号不切。"""
    out: List[str] = []
    buf: List[str] = []
    quote: Optional[str] = None
    depth = 0
    for ch in s:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            buf.append(ch)
        elif ch in "([{":
            depth += 1
            buf.append(ch)
        elif ch in ")]}":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if "".join(buf).strip():
        out.append("".join(buf).strip())
    return [x for x in out if x]


def _strip_quotes(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    return v


def parse_kwargs(arg_str: str) -> Dict[str, str]:
    """把 ``a=1, b="x, y"`` 解析成 dict。值一律先当字符串，交给 call_tool 按注解转换。"""
    args: Dict[str, str] = {}
    for part in _split_top_level(arg_str):
        if "=" not in part:
            raise ParseError(f"参数 `{part}` 不是 k=v 形式")
        k, v = part.split("=", 1)
        args[k.strip()] = _strip_quotes(v)
    return args


def parse_action(text: str) -> Tuple[str, str, Dict[str, str]]:
    """解析一轮模型输出，返回 ``(thought, tool_name, kwargs)``。

    ``tool_name == "finish"`` 表示结束；kwargs 里带 ``conclusion``。
    """
    m = _THOUGHT_RE.search(text)
    thought = m.group(1).strip() if m else ""

    m = _ACTION_RE.search(text)
    if m:
        call = m.group(1).strip().rstrip("。.")
        # 去掉 ``` 包裹
        call = call.strip("`")
        if "(" in call and call.rstrip().endswith(")"):
            name = call[: call.index("(")].strip()
            inner = call[call.index("(") + 1 : call.rfind(")")]
            return thought, name, parse_kwargs(inner)
        return thought, call, {}

    m = _FINAL_RE.search(text)
    if m:
        return thought, "finish", {"conclusion": m.group(1).strip()}

    raise ParseError(
        "没找到 Action。请严格按格式输出：\nThought: ...\nAction: 工具名(参数=值)"
    )


# ---------------------------------------------------------------- 工具调用


def _coerce(value: str, annotation: Any) -> Any:
    """按工具函数签名把字符串参数转成正确的 Python 类型。

    工具内部虽然大多也会自己转，但在**调用前**统一转换可以让错误更早暴露，
    也让轨迹里的 payload 类型稳定（评测要做数值比较）。
    """
    ann = str(annotation)
    if ann in ("<class 'NoneType'>", "inspect._empty", ""):
        return value
    try:
        if "int" in ann:
            return int(float(value))
        if "float" in ann:
            return float(value)
        if "bool" in ann:
            return str(value).strip().lower() in ("1", "true", "yes", "on")
    except (TypeError, ValueError) as exc:
        raise ParseError(f"参数值 `{value}` 无法转成 {ann}：{exc}") from exc
    return value


def call_tool(tool: Tool, kwargs: Dict[str, str]) -> Tuple[ToolResult, Dict[str, Any]]:
    """调用工具，按签名做参数过滤与类型转换。

    返回 ``(result, resolved_args)``。未知参数**静默丢弃**并在文本里提示，
    免得模型因为多写了一个参数就整轮失败。
    """
    sig = inspect.signature(tool.func.func)  # functools.partial 包装
    params = {}
    for name, p in sig.parameters.items():
        if name in ("kw", "kwargs") and p.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        params[name] = p.annotation

    resolved: Dict[str, Any] = {}
    unknown = [k for k in kwargs if k not in params]
    for k, v in kwargs.items():
        if k in params:
            resolved[k] = _coerce(v, params[k])

    res = tool.func(**resolved)
    if unknown and res.ok:
        res.text += f"\n（忽略未知参数：{', '.join(unknown)}）"
    return res, resolved


# ---------------------------------------------------------------- 轨迹


@dataclass
class Step:
    index: int
    thought: str
    action: str
    args: Dict[str, Any]
    observation: str
    ok: bool
    payload: Dict[str, Any] = field(default_factory=dict)
    raw: str = ""


@dataclass
class RunResult:
    task: str
    steps: List[Step] = field(default_factory=list)
    conclusion: str = ""
    stopped_reason: str = ""      # finish / max_steps / parse_error_exhausted / error
    tool_calls: List[str] = field(default_factory=list)
    backend: str = ""
    strict: bool = True
    honest: bool = True
    with_permutation: bool = True
    error: str = ""

    # ---------------------------------------------------------- 评测用快捷查询

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def finished(self) -> bool:
        return self.stopped_reason == "finish"

    def called(self, name: str) -> bool:
        return name in self.tool_calls

    def payload_of(self, name: str) -> Dict[str, Any]:
        """取某个工具最后一次成功调用的结构化结果（评测要读 p 值、AUC）。"""
        for s in reversed(self.steps):
            if s.action == name and s.ok:
                return s.payload
        return {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "conclusion": self.conclusion,
            "stopped_reason": self.stopped_reason,
            "backend": self.backend,
            "ablation": {
                "strict": self.strict,
                "honest": self.honest,
                "with_permutation": self.with_permutation,
            },
            "tool_calls": self.tool_calls,
            "n_steps": self.n_steps,
            "finished": self.finished,
            "error": self.error,
            "steps": [
                {
                    "i": s.index,
                    "thought": s.thought,
                    "action": s.action,
                    "args": s.args,
                    "ok": s.ok,
                    "observation": s.observation,
                }
                for s in self.steps
            ],
        }


# ---------------------------------------------------------------- 循环


def run_react(
    session: AnalysisSession,
    llm: LLM,
    task: str = "判断该数据集中感染组与健康组是否存在可区分的代谢信号",
    data_dir: str = "",
    max_steps: int = 12,
    honest: bool = True,
    with_permutation: bool = True,
    strict: bool = True,
    max_parse_retries: int = 2,
    verbose: bool = False,
) -> RunResult:
    """跑一轮 ReAct。

    - ``honest`` / ``with_permutation`` 控制**工具层**口径（见 tools.build_tools）；
    - ``strict`` 控制**提示词层**口径（见 prompts.build_system_prompt）。
      两者独立，正是消融设计想要的。
    """
    tools = build_tools(session, honest=honest, with_permutation=with_permutation)
    by_name = {t.name: t for t in tools}

    system = build_system_prompt(tools, strict=strict)
    messages: List[Message] = [
        {"role": "system", "content": system},
        {"role": "user", "content": build_user_prompt(task, data_dir)},
    ]

    run = RunResult(
        task=task, backend=llm.name, strict=strict,
        honest=honest, with_permutation=with_permutation,
    )

    parse_fail_streak = 0
    for i in range(max_steps):
        try:
            raw = llm.complete(messages)
        except Exception as exc:  # noqa: BLE001
            run.stopped_reason = "error"
            run.error = f"LLM 调用失败：{exc}"
            break

        try:
            thought, name, kwargs = parse_action(raw)
        except ParseError as exc:
            parse_fail_streak += 1
            run.steps.append(
                Step(i, "", "<parse_error>", {}, str(exc), False, raw=raw)
            )
            if parse_fail_streak > max_parse_retries:
                run.stopped_reason = "parse_error_exhausted"
                run.error = f"连续 {parse_fail_streak} 轮无法解析模型输出"
                break
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {"role": "user", "content": f"Observation: [格式错误] {exc}"}
            )
            continue

        parse_fail_streak = 0
        tool = by_name.get(name)
        if tool is None:
            obs = (
                f"[错误] 没有名为 `{name}` 的工具。可用工具："
                + ", ".join(by_name.keys())
            )
            run.steps.append(Step(i, thought, name, dict(kwargs), obs, False, raw=raw))
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Observation: {obs}"})
            continue

        try:
            res, resolved = call_tool(tool, kwargs)
        except ParseError as exc:
            res, resolved = ToolResult(f"[错误] {exc}", {"error": str(exc)}), dict(kwargs)
        except Exception as exc:  # noqa: BLE001
            run.stopped_reason = "error"
            run.error = f"工具 {name} 抛异常：{exc}"
            break

        run.steps.append(
            Step(i, thought, name, resolved, res.text, res.ok, res.payload, raw=raw)
        )
        run.tool_calls.append(name)

        if verbose:
            print(f"[{i}] {thought}\n    → {name}({resolved})\n    {res.text}\n")

        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": f"Observation: {res.text}"})

        if name == "finish" and res.ok:
            run.conclusion = res.payload.get("conclusion", "")
            run.stopped_reason = "finish"
            break
    else:
        run.stopped_reason = "max_steps"
        run.error = f"达到步数上限 {max_steps}，未给出结论"

    return run


__all__ = [
    "Step",
    "RunResult",
    "ParseError",
    "parse_action",
    "parse_kwargs",
    "call_tool",
    "run_react",
]
