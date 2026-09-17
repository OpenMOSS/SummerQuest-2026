"""python student_scripts/a2k/summarize_flash_tests.py \\
    --run "python -m pytest tests/test_attention.py -v" \\
    --gpu "NVIDIA GeForce RTX 4090"
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path("local_results/a2k/pytest")
REPORT_FILENAME = "unit_tests.txt"
MAX_OUTPUT_BYTES = 5 * 1024 * 1024
STARTER_COMMIT = "ca8bc81a59b70516f7ebb2da4808daade877c736"

_CLUSTER_PREFIX = "/remote-home\\d+"
REDACTION_RULES: tuple[tuple[str, str], ...] = (
    (rf"({_CLUSTER_PREFIX})/[^/\s]+", r"\1/<user>"),
    (r"/home/[^/\s]+", "<home>"),
    (r"/Users/[^/\s]+", "<home>"),
    (r"/var/tmp/[^\s:]*", "<tmp>"),
    (r"/tmp/[^\s:]*", "<tmp>"),
    (r"/scratch/[^\s:]*", "<scratch>"),
    (r"/lustre/[^\s:]*", "<lustre>"),
    (r"[A-Za-z]:\\\\+Users\\\\+[^\\\s]+", "<home>"),
    (r"[A-Za-z]:\\+Users\\+[^\\\s]+", "<home>"),
    (r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "<ip>"),
    (r"\b192\.168\.\d{1,3}\.\d{1,3}\b", "<ip>"),
    (r"\b172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b", "<ip>"),
    (r"\b[a-zA-Z0-9._-]+\.(?:local|cluster|internal)\b", "<host>"),
)

_PLACEHOLDERS = ("<workspace>", "<home>", "<tmp>", "<scratch>", "<lustre>", "<ip>", "<host>", "<user>")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", metavar="COMMAND", required=True, help="要执行的 pytest 命令（在已申请到单张 RTX 4090 的进程里运行）")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="报告输出目录")
    parser.add_argument("--output-name", default=REPORT_FILENAME, help="报告文件名")
    parser.add_argument("--env", type=Path, default=None, help="环境 JSON，用于自动填写 GPU 型号")
    parser.add_argument("--gpu", default=None, help="手工指定 GPU 型号")
    parser.add_argument("--commit", default=None, help="测试代码对应的 commit")
    parser.add_argument("--repo", default=".", help="用于读取 commit 的仓库根目录")
    parser.add_argument("--timeout", type=int, default=1800, help="--run 的超时秒数")
    parser.add_argument("--allow-no-pass", action="store_true", help="允许没有任何用例通过时也出报告")
    return parser.parse_args(argv)


def redact(text: str) -> str:
    for pattern, replacement in REDACTION_RULES:
        text = re.sub(pattern, replacement, text)
    return text


def usernames_in(text: str) -> list[str]:
    found: list[str] = []
    for pattern in (r"/remote-home\d+/([^/\s]+)", r"/home/([^/\s]+)", r"/Users/([^/\s]+)"):
        for match in re.finditer(pattern, text):
            name = match.group(1)
            if not name.startswith("<") and name not in found:
                found.append(name)
    return found


def run_pytest(command: str, timeout: int) -> tuple[str, int]:
    with tempfile.NamedTemporaryFile("w+", suffix=".log", delete=True, encoding="utf-8") as handle:
        completed = subprocess.run(
            shlex.split(command), stdout=handle, stderr=subprocess.STDOUT, timeout=timeout, check=False
        )
        handle.flush()
        handle.seek(0)
        return handle.read(), completed.returncode


def parse_summary(text: str) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "xpassed": 0, "errors": 0}
    for number, label in re.findall(r"(\d+)\s+(passed|failed|skipped|xfailed|xpassed|error|errors)", text):
        key = "errors" if label in {"error", "errors"} else label
        counts[key] += int(number)
    return counts


def parse_outcomes(text: str) -> dict[str, list[str]]:
    outcomes: dict[str, list[str]] = {"passed": [], "failed": [], "skipped": [], "error": []}
    for match in re.finditer(
        r"^(\S*test_\S*?::\S+)\s+(PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)\b", text, flags=re.MULTILINE
    ):
        name, status = match.group(1), match.group(2).lower()
        key = {"xfail": "skipped", "xpass": "passed"}.get(status, status)
        if name not in outcomes.get(key, []):
            outcomes.setdefault(key, []).append(name)
    for match in re.finditer(r"^(FAILED|ERROR)\s+(\S+?::\S+)", text, flags=re.MULTILINE):
        status, name = match.group(1).lower(), match.group(2)
        for bucket in outcomes.values():
            if name in bucket:
                bucket.remove(name)
        outcomes.setdefault(status, []).append(name)
    return outcomes


def triton_skips(outcomes: dict[str, list[str]]) -> list[str]:
    return [name for name in outcomes.get("skipped", []) if "triton" in name.lower()]


def resolve_commit(repo: Path, override: str | None) -> str:
    if override:
        return override
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        value = completed.stdout.strip()
        if value:
            return value
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return STARTER_COMMIT


def read_environment(env_path: Path | None) -> dict:
    if env_path is None or not env_path.is_file():
        return {}
    try:
        payload = json.loads(env_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def describe_gpu(environment: dict, override: str | None) -> str:
    if override:
        return override
    gpu = (environment.get("gpu") or {}) if environment else {}
    if gpu.get("name"):
        detail = gpu["name"]
        if gpu.get("total_memory_mib"):
            detail += f" ({gpu['total_memory_mib']} MiB)"
        return detail
    return "未提供"


def build_report(
    *,
    counts: dict[str, int],
    outcomes: dict[str, list[str]],
    triton_skipped: list[str],
    gpu: str,
    command: str,
    commit: str,
    environment: dict,
) -> str:
    lines: list[str] = [
        "A2-K FlashAttention 测试报告",
        "",
        "覆盖范围：任务三前向（PyTorch tiled + Triton）+ 任务四反向（重计算）。",
        "判定口径：skipped 不计入 passed；没有 CUDA 时被 skip 的 Triton 用例仍标记为 skipped。",
        "",
        "运行信息：",
        f"  GPU: {gpu}",
        f"  Command: {command}",
        f"  Commit: {commit}",
    ]

    if environment:
        env_bits = [
            f"{key}={environment[key]}"
            for key in ("python", "torch", "torch_cuda", "triton", "cudnn")
            if environment.get(key)
        ]
        if env_bits:
            lines.append("  环境: " + ", ".join(env_bits))
        if environment.get("nvidia_smi"):
            smi_bits = [f"{k}={v}" for k, v in environment["nvidia_smi"].items() if v is not None]
            if smi_bits:
                lines.append("  nvidia-smi: " + ", ".join(smi_bits))

    lines += [
        "",
        "统计：",
        f"  passed:  {counts['passed']}",
        f"  failed:  {counts['failed']}",
        f"  skipped: {counts['skipped']}",
        f"  xfailed: {counts['xfailed']}",
        f"  xpassed: {counts['xpassed']}",
        f"  errors:  {counts['errors']}",
        f"  合计:    {sum(counts.values())}",
    ]
    if triton_skipped:
        lines.append(f"  被跳过的 Triton 用例 {len(triton_skipped)} 个：未在 CUDA 上验证，不能算作通过。")

    lines += ["", "逐用例状态："]
    label_map = (("passed", "PASS"), ("failed", "FAIL"), ("skipped", "SKIP"), ("error", "ERROR"))
    listed = 0
    for key, label in label_map:
        for name in outcomes.get(key, []):
            lines.append(f"  [{label}] {name}")
            listed += 1
    if listed == 0:
        lines.append("  （未能逐条解析用例状态）")

    return "\n".join(lines) + "\n"


def find_unredacted(text: str, usernames: list[str] | None = None) -> list[str]:
    hits: list[str] = []
    for name in usernames or []:
        if name and name in text:
            hits.append(f"用户名残留: {name}")
    patterns = [
        r"/remote-home\d+/\S*",
        r"/home/\S*",
        r"/Users/\S*",
        r"/tmp/\S*",
        r"/var/tmp/\S*",
        r"/scratch/\S*",
        r"/lustre/\S*",
        r"[A-Za-z]:\\*Users\\*\S*",
        r"\b10\.\d+\.\d+\.\d+\b",
        r"\b192\.168\.\d+\.\d+\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            chunk = match.group(0)
            if not any(placeholder in chunk for placeholder in _PLACEHOLDERS):
                hits.append(chunk)
    return hits


def write_report(text: str, target: Path) -> int:
    size = len(text.encode("utf-8"))
    if size > MAX_OUTPUT_BYTES:
        print(
            f"警告: {target} 为 {size / 1024**2:.2f} MiB，超过单文件上限 "
            f"{MAX_OUTPUT_BYTES / 1024**2:.0f} MiB。",
            file=sys.stderr,
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return size


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = Path(args.repo).resolve()

    print(f"执行: {args.run}")
    try:
        raw_text, returncode = run_pytest(args.run, args.timeout)
    except subprocess.TimeoutExpired:
        print(f"错误: 命令超过 {args.timeout} 秒未结束。", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"错误: 找不到可执行文件 —— {exc}", file=sys.stderr)
        return 2
    if returncode != 0:
        print(f"注意: 命令退出码 {returncode}（pytest 有失败时为 1）。", file=sys.stderr)

    redacted = redact(raw_text)
    counts = parse_summary(redacted)
    outcomes = parse_outcomes(redacted)
    triton_skipped = triton_skips(outcomes)

    if counts["passed"] == 0 and counts["skipped"] > 0 and not args.allow_no_pass:
        print(
            f"错误: passed=0 而 skipped={counts['skipped']}，说明没有用例真正通过。"
            "确需保留时加 --allow-no-pass。",
            file=sys.stderr,
        )
        return 1

    environment = read_environment(args.env)
    report = build_report(
        counts=counts,
        outcomes=outcomes,
        triton_skipped=triton_skipped,
        gpu=describe_gpu(environment, args.gpu),
        command=redact(args.run),
        commit=resolve_commit(repo, args.commit),
        environment=environment,
    )

    target = args.output_dir / args.output_name
    size = write_report(report, target)

    print(f"报告已写入 {target}  ({size / 1024:.1f} KiB，已脱敏)")
    print(
        f"  统计: passed={counts['passed']} failed={counts['failed']} "
        f"skipped={counts['skipped']} errors={counts['errors']}"
    )
    if triton_skipped:
        print(f"  被跳过的 Triton 用例 {len(triton_skipped)} 个 —— 未在 CUDA 上验证，不能算作通过。")
    if counts["failed"]:
        print(f"  注意: 有 {counts['failed']} 个用例失败，需先修复。", file=sys.stderr)

    leftover = find_unredacted(report, usernames_in(args.run))
    if leftover:
        print(f"  警告: 输出中可能仍有未脱敏的内容: {sorted(set(leftover))[:5]}", file=sys.stderr)
    else:
        print("  脱敏自查: 未发现残留的用户名 / 内部路径 / IP")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
