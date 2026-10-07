"""
合成对话数据生成脚本（data_process/generate_synthetic_data.py）

流程（每一条样本）：
1. 批量问题：生产者线程用一批对话让模型基于随机主题生成 QUESTIONS_PER_BATCH 个
   问题（一次拿一整批，减少请求次数），随后按编号解析并逐条校验；
   QUESTIONS_PER_BATCH=1 时自动取消批量机制（单题模式：每次对话只出一题，
   不做编号解析，直接校验入队）
2. 去重：规范化后与历史已生成问题比对，重复则丢弃（可跨运行去重；除当前输出
   jsonl 外，还会自动合并旧版产物 data_process\\synthetic_sft.jsonl）
3. 新对话回答：每个问题单独送入全新上下文生成回答（避免"接着编"）
4. 校验 + 输出：一行一个样本，样本内换行转义为字面 \\n：
   "<|im_start|>user\\n问题<|im_end|>\\n<|im_start|>assistant\\n回答<|im_end|>\\n"
5. 断点续跑：启动时从已有 jsonl 载入去重集合，中断后重跑只会补新样本

并发（生产者-消费者流水线，替代旧版"阶段 A 出题→屏障→阶段 C 回答"）：
- 启动单个 llama-server，一个模型实例同时服务多路独立对话，避免每样本重复加载模型
- 线程构成：P 个出题生产者（--producers，默认 1）+ N 个回答消费者（--parallel）
  + 主线程按序写盘
- 服务端槽位 -np（--slots，默认 = P + N 与线程 1:1）；手动调小则多出的请求在
  服务端排队。注意 llama.cpp 把 -c 总上下文均摊给所有槽位，槽位越多单槽上下文
  越小、长回答会被截断，因此 -c 按槽位数自动预留（--context 0 = 自动）
- 批量模式 1 个生产者通常够用（一次出题请求能补一整批）；单题模式（qpb=1）
  一个请求只出一题，生产者喂不饱消费者时按需调大 --producers
- 出题→去重→入有界缓冲（--buffer，默认一整批）→回答→结果队列→写盘，各级
  解耦：缓冲满则生产者自然节流，回答一完成就写盘，不再互相等整批对齐
- 达到 --count 后置位 stop_event，生产者停发、消费者排空退出；生产者连续
  MAX_EMPTY_ROUNDS 轮产不出新问题（服务异常/去重全命中）则终止报错，不再无限重试

用法：
    uv run python data_process/generate_synthetic_data.py --count 50 --parallel 8
    uv run python data_process/generate_synthetic_data.py --topics 历史,科学 --count 20 --temp 1.0
"""

import argparse
import json
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque

# ========================= 可配置常量 =========================
# 以下为未指定 CLI 参数时的默认值；日常调参直接走命令行（如 --temp 0.9 --max-tokens 512）

# llama.cpp 二进制目录（含 llama-server.exe）
LLAMA_BIN_DIR = r"D:\llama-b10472-bin-win-cuda-13.3-x64"
LLAMA_SERVER_EXE = os.path.join(LLAMA_BIN_DIR, "llama-server.exe")

# 模型路径（GGUF）
MODEL_PATH = r"D:\Model\Qwen3.5-4B-UD-Q4_K_XL.gguf"

# 输出文件（ChatML 每行一条对话；jsonl 含元数据、用于去重和断点续跑）
OUTPUT_TXT = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "train_text", "SFT", "synthetic_sft.txt")
OUTPUT_JSONL = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "train_text", "SFT", "synthetic_sft.jsonl")
# 旧版脚本曾把 jsonl 写在脚本目录；存在且与 --output-jsonl 不同时一并载入去重集合，
# 避免跨运行去重断链
LEGACY_JSONL = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "synthetic_sft.jsonl")
SERVER_LOG = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "llama_server.log")

# llama-server 配置
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 18080
PARALLEL = 16                # --parallel：回答消费者线程数。2026-10-06 完整扫描+
                             # 服务端探针实测（off+fa/poll）：聚合 tok/s
                             # np6=239 / np12=338 / np16=394，K 超订阅仅 +8%；
                             # 16 为吞吐最优，再高进入平台期（每步固定成本墙）
PRODUCERS = 1                # --producers：出题生产者线程数（实测恒为 1 最优：
                             # 单题模式 P2N2s4≈19 条/分 vs P1N4s4≈43，多出的生产者
                             # 过度出题、在途请求偷走回答槽位算力，不要调大）
SLOTS = 0                    # -np：服务端槽位数；0 = 自动（生产者 + 消费者）
# -c：总上下文 token；0 = 自动（槽数 × (max_tokens + 256)）
CONTEXT = 0
SERVER_READY_TIMEOUT = 120   # 等待服务就绪（秒）

# 生成参数
NUM_SAMPLES = 1000           # 本次要生成的新样本数（去重后净增）
MAX_N_TOKENS = 768           # 单次生成最大 token 数（--max-tokens 可覆盖）
GPU_LAYERS = 99              # 送入 GPU 的层数（-ngl）
SPEC_DRAFT_N_MAX = 0         # MTP 投机深度；0=关闭。实测高并行下投机净亏：
                             # 探针 np12 贪心 mtp2=308.6 vs off=337.9 tok/s(×0.91)，
                             # 流水线 t768 off 比 mtp2 条/分 +19%（低 N≤2 才占优）
TEMPERATURE = 0.7            # 采样温度（--temp 可覆盖）
SEED = -1                    # 随机种子（-1 = 每次随机）
MAX_RETRIES = 3              # 单条样本失败重试次数
REQUEST_TIMEOUT = 300        # 单次 HTTP 请求超时（秒）
QUESTIONS_PER_BATCH = 16     # 一个对话批量生成的问题数（=1 时自动取消批量机制）
MAX_EMPTY_ROUNDS = 8         # 生产者连续这么多轮零产新题则判服务异常并终止

# 问题生成的主题池（--topics 可覆盖）
TOPICS = [
    "中国历史", "日常生活", "日常对话", "科学常识", "常识问答", "编程技术", "文学创作",
    "心理健康", "学习方法", "数学", "科技数码", "自我介绍与身份问答",
    "美食", "旅行", "职业规划", "传统文化", "人工智能",
]

# 系统提示词
SYSTEM_QUESTION_PROMPT = (
    "你是一个中文数据生成助手。请基于给定的主题，生成一批互不相同、**答案简单**、"
    "有讨论价值的问题。每行一个问题，用\"1.\" \"2.\"等数字编号开头。"
    "只输出编号问题列表，不要任何解释、前缀或后缀。"
)
# 单题模式（QUESTIONS_PER_BATCH=1）专用：一次对话只出一题
SYSTEM_SINGLE_QUESTION_PROMPT = (
    "你是一个中文数据生成助手。请基于给定的主题，生成一个答案简单、"
    "有讨论价值的简单问题。只输出问题本身一行文字，"
    "不要编号、引号、解释或任何前后缀。"
)
SYSTEM_ANSWER_PROMPT = (
    "你是一个乐于助人的中文AI助手。请认真回答用户的问题，"
    "给出简要的回答，不超过400字。"
)

# 问题质量黑名单（命中即重试）：模型常见"解释自己设计"的废话
QUESTION_BLACKLIST = [
    "设计思路", "以下是", "以上是", "这个问题", "希望对你有帮助",
    "如果你愿意", "我们来", "我设计了", "本题设计",
]

MIN_Q_LEN = 6      # 问题最短字符数
MAX_Q_LEN = 100    # 问题最长字符数
MIN_A_LEN = 10     # 回答最短字符数

# ========================= 基础工具 =========================


def normalize(text: str) -> str:
    """规范化文本用于去重：去空白与标点、统一小写。"""
    text = text.lower()
    text = re.sub(
        r"[\s\u3000，。！？、；：,.!?;:()（）\[\]【】\"'“”‘’·…\-—_~*#]+", "", text)
    return text.strip()


def sanitize(text: str) -> str:
    """清除回复中可能残留的 ChatML 角色标记。"""
    text = re.sub(r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>", "", text)
    text = re.sub(
        r"^\s*(assistant|user|system)\s*[:：]", "", text, flags=re.MULTILINE)
    return text.strip()


def to_chatml_line(question: str, answer: str) -> str:
    """一行一个样本（不含末尾换行），样本内所有换行转义为字面 \\n：
    "<|im_start|>user\\n问题<|im_end|>\\n<|im_start|>assistant\\n回答<|im_end|>\\n"
    （对齐 范文集_script.py 的 format_as_line 行式约定）
    写入时需自行追加真实换行 \\n 作为样本间分隔。"""
    q = question.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
    a = answer.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
    return f"<|im_start|>user\\n{q}<|im_end|>\\n<|im_start|>assistant\\n{a}<|im_end|>\\n"


def validate_question(q: str) -> bool:
    qn = normalize(q)
    if len(qn) < MIN_Q_LEN or len(q) > MAX_Q_LEN:
        return False
    return not any(w in q for w in QUESTION_BLACKLIST)


def validate_answer(a: str, qn: str) -> bool:
    an = normalize(a)
    if len(an) < MIN_A_LEN:
        return False
    return an != qn  # 回答不能只是复述问题


def load_seen(*paths: str) -> set:
    """载入历史问题的规范化集合（断点续跑去重）。

    可传入多个路径（当前输出 + 旧版产物）；损坏行容错跳过。"""
    seen = set()
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["q_norm"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return seen


# ========================= llama-server 管理 =========================

class LlamaServer:
    """管理 llama-server 进程：启动（或复用已有实例）、健康检查、请求、关闭。"""

    def __init__(self, port: int, parallel: int, gpu_layers: int,
                 context: int):
        self.port = port
        self.parallel = parallel      # -np：槽位数
        self.gpu_layers = gpu_layers
        self.context = context        # -c：总上下文（均摊给所有槽位）
        self.proc = None
        self.logf = None
        self.started_here = False

    def start(self) -> None:
        """启动服务（若端口已有健康实例则直接复用）。"""
        if self._health_ok():
            print(f"[info] 复用端口 {self.port} 上已运行的 llama-server")
            return
        cmd = [
            LLAMA_SERVER_EXE,
            "-m", MODEL_PATH,
            "-ngl", str(self.gpu_layers),
            "-c", str(self.context),
            "-np", str(self.parallel),
            "--host", SERVER_HOST,
            "--port", str(self.port),
            # 服务端 A/B 实测：flash-attn + poll 等待使聚合解码 +9.6%
            # （探针：同 prompt 贪心 K=16，360.0→394.5 tok/s @ np16）
            "--flash-attn", "on",
            "--poll", "1",
            # 服务级禁用 Qwen 思考模式，所有请求生效
            "--chat-template-kwargs", '{"enable_thinking": false}',
            "--reasoning", "off",
            # 禁用 prompt cache（cache-ram 默认 8GB）：每次请求都是全新对话，
            # 缓存无法复用，实测内存会随请求数线性增长到 8GB 上限
            "--cache-ram", "0",
        ]
        if SPEC_DRAFT_N_MAX > 0:
            # MTP 投机解码；--spec-draft-n-max 只接受整数（"auto" 会让服务启动即崩）
            cmd += ["--spec_type", "draft-mtp",
                    "--spec-draft-n-max", str(SPEC_DRAFT_N_MAX)]
        self.logf = open(SERVER_LOG, "a", encoding="utf-8")
        self.proc = subprocess.Popen(
            cmd, stdout=self.logf, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        self.started_here = True
        if not self.wait_ready():
            raise RuntimeError(
                f"llama-server 启动失败，请查看日志: {SERVER_LOG}")
        print(f"[info] llama-server 已启动: {self.parallel} 槽位并发")

    def _health_ok(self) -> bool:
        try:
            with urllib.request.urlopen(
                    f"http://{SERVER_HOST}:{self.port}/health", timeout=2) as r:
                return r.status == 200 and "ok" in r.read().decode("utf-8", "replace")
        except Exception:
            return False

    def wait_ready(self, timeout: int = SERVER_READY_TIMEOUT) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                return False  # 进程已退出
            if self._health_ok():
                return True
            time.sleep(0.5)
        return False

    def chat(self, messages: list, seed: int, max_tokens: int,
             temperature: float) -> str:
        """OpenAI 兼容接口 /v1/chat/completions，返回 assistant 文本。

        max_tokens / temperature 由调用方显式传入（来自 CLI 参数），
        不要在这里引用模块常量，否则 --max-tokens/--temp 会变成死参数。"""
        body = json.dumps({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://{SERVER_HOST}:{self.port}/v1/chat/completions",
            data=body, headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
            resp = json.loads(r.read().decode("utf-8"))
        content = resp["choices"][0]["message"].get("content") or ""
        return content.strip()

    def close(self) -> None:
        if self.started_here and self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            print("[info] llama-server 已关闭")
        if self.logf is not None:
            self.logf.close()


# ========================= 生成步骤（worker 内调用） =========================

def generate_question_batch(server: LlamaServer, topic: str, seed: int,
                            n_questions: int, temperature: float,
                            max_tokens: int) -> list[str]:
    """步骤 1：一个对话批量生成 n_questions 个问题（重试，全部失败返回 []）。"""
    messages = [
        {"role": "system", "content": SYSTEM_QUESTION_PROMPT},
        {"role": "user",
         "content": f"请基于主题「{topic}」生成 {n_questions} 个不同的问题。"},
    ]
    for attempt in range(MAX_RETRIES):
        try:
            raw = sanitize(server.chat(messages, seed + attempt, max_tokens,
                                       temperature))
            qs = [q for q in parse_question_batch(raw) if validate_question(q)]
        except Exception as e:
            print(f"  [warn] 问题批量生成失败({e})，重试 {attempt + 1}/{MAX_RETRIES}")
            continue
        if len(qs) >= 2:
            return qs
        print(f"  [warn] 问题解析不足({len(qs)}个)，重试 {attempt + 1}/{MAX_RETRIES}")
    return []


def parse_question_batch(text: str) -> list[str]:
    """解析编号问题列表："1. xxx\n2. yyy" → ["xxx", "yyy"]（支持问题跨多行）。"""
    questions: list[str] = []
    cur: list[str] = []
    for line in text.splitlines():
        m = re.match(r"^\s*\d+[.、)）]\s*(.+)$", line)
        if m:
            if cur:
                questions.append(" ".join(cur).strip())
            cur = [m.group(1)]
        elif cur:
            cur.append(line.strip())
    if cur:
        questions.append(" ".join(cur).strip())
    return [q for q in questions if q]


def parse_single_question(text: str) -> str:
    """单题模式解析：取第一个非空行，剥离模型可能自带的编号前缀与包裹引号。"""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\d+[.、)）]\s*(.+)$", line)
        if m:
            line = m.group(1).strip()
        return line.strip("\"'“”‘’《》 ")
    return ""


def generate_single_question(server: LlamaServer, topic: str, seed: int,
                             temperature: float, max_tokens: int) -> str | None:
    """步骤 1（单题模式，QUESTIONS_PER_BATCH=1）：一次对话只生成一个问题。

    不经过编号列表解析，直接校验；重试全部失败返回 None。"""
    messages = [
        {"role": "system", "content": SYSTEM_SINGLE_QUESTION_PROMPT},
        {"role": "user", "content": f"请基于主题「{topic}」提出一个问题。"},
    ]
    for attempt in range(MAX_RETRIES):
        try:
            raw = sanitize(server.chat(messages, seed + attempt, max_tokens,
                                       temperature))
            q = parse_single_question(raw)
        except Exception as e:
            print(f"  [warn] 单题生成失败({e})，重试 {attempt + 1}/{MAX_RETRIES}")
            continue
        if q and validate_question(q):
            return q
        print(f"  [warn] 单题解析无效，重试 {attempt + 1}/{MAX_RETRIES}")
    return None


def generate_answer(server: LlamaServer, question: str, seed: int,
                    temperature: float, max_tokens: int) -> str | None:
    """步骤 3：把问题送入全新对话生成回答（内部重试，全部失败返回 None）。"""
    messages = [
        {"role": "system", "content": SYSTEM_ANSWER_PROMPT},
        {"role": "user", "content": question},
    ]
    for attempt in range(MAX_RETRIES):
        try:
            a = sanitize(server.chat(messages, seed + attempt + 1000,
                                     max_tokens, temperature))
        except Exception as e:
            print(f"  [warn] 回答生成失败({e})，重试 {attempt + 1}/{MAX_RETRIES}")
            continue
        if validate_answer(a, normalize(question)):
            return a
    return None


# ========================= 流水线线程（生产者 / 消费者） =========================

def produce_questions(server: LlamaServer, args, topics: list, seen: set,
                      seen_lock: threading.Lock, q_buf: queue.Queue,
                      stop_event: threading.Event,
                      producer_id: int = 0) -> None:
    """生产者（可同时跑多个，用 producer_id 区分）：循环"出题（批量或单题）→ 去重 → 入缓冲"。

    每个生产者持有独立 Random（seed 按 producer_id 偏移），避免多线程用相同
    主题/采样种子重复出题；共享去重集合由 seen_lock 保护。
    缓冲满时阻塞在 put 上（天然节流，与回答/写盘速度解耦）；
    连续 MAX_EMPTY_ROUNDS 轮拿不到新问题（服务异常 / 去重全命中）则置位
    stop_event 退出，避免无限空转。"""
    rng = random.Random(
        None if args.seed < 0 else args.seed + 7919 * producer_id)
    empty_rounds = 0
    while not stop_event.is_set():
        topic = rng.choice(topics)
        seed = rng.randint(0, 2**31 - 1)
        if args.questions_per_batch <= 1:
            q = generate_single_question(server, topic, seed,
                                         args.temp, args.max_tokens)
            qs = [q] if q else []
        else:
            qs = generate_question_batch(server, topic, seed,
                                         args.questions_per_batch,
                                         args.temp, args.max_tokens)
        new_items = []
        with seen_lock:
            for q in qs:
                qn = normalize(q)
                if qn in seen:
                    print(f"  [dup] 问题已存在，跳过: {q[:50]}")
                    continue
                seen.add(qn)
                new_items.append((q, qn, topic))
        if not new_items:
            empty_rounds += 1
            if empty_rounds >= MAX_EMPTY_ROUNDS:
                print(f"[error] 连续 {MAX_EMPTY_ROUNDS} 轮未产出新问题"
                      f"（服务异常或去重全命中），停止生产")
                stop_event.set()
                return
            continue
        empty_rounds = 0
        for item in new_items:
            while not stop_event.is_set():
                try:
                    q_buf.put(item, timeout=0.5)
                    break
                except queue.Full:
                    pass  # 缓冲满，等消费者腾出空间（每 0.5s 重查停止信号）


def answer_worker(server: LlamaServer, args, worker_id: int, q_buf: queue.Queue,
                  results: queue.Queue, stop_event: threading.Event) -> None:
    """消费者：从缓冲取问题，用全新对话生成回答，结果交主线程按序写盘。

    每个在途请求独占一个服务端槽位；stop_event 置位后（配额已满或生产终止）
    放弃剩余问题直接退出。"""
    rng = random.Random(
        None if args.seed < 0 else args.seed + 1000 + worker_id)
    while not stop_event.is_set():
        try:
            q, qn, topic = q_buf.get(timeout=0.5)
        except queue.Empty:
            continue
        if stop_event.is_set():
            return
        a = generate_answer(server, q, rng.randint(0, 2**31 - 1),
                            args.temp, args.max_tokens)
        results.put((q, qn, topic, a))


# ========================= 主流程 =========================

def build_parser() -> argparse.ArgumentParser:
    """独立成函数，便于基准测试等脚本复用同一套参数定义。"""
    parser = argparse.ArgumentParser(
        description="用 llama.cpp 生产者-消费者流水线生成合成 Q&A 数据")
    parser.add_argument("--count", type=int,
                        default=NUM_SAMPLES, help="生成样本数（去重后净增）")
    parser.add_argument("--topics", type=str, default=None,
                        help="逗号分隔的主题列表，覆盖 TOPICS")
    parser.add_argument("--output-txt", type=str, default=OUTPUT_TXT)
    parser.add_argument("--output-jsonl", type=str, default=OUTPUT_JSONL)
    parser.add_argument("--gpu-layers", type=int, default=GPU_LAYERS)
    parser.add_argument("--temp", type=float, default=TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=MAX_N_TOKENS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--parallel", type=int,
                        default=PARALLEL,
                        help="回答消费者线程数（出题生产者由 --producers 单独设置）")
    parser.add_argument("--producers", type=int, default=PRODUCERS,
                        help="出题生产者线程数；默认槽位数 = 生产者 + 消费者")
    parser.add_argument("--slots", type=int, default=SLOTS,
                        help="llama-server 槽位数（-np）；0 = 自动，小于线程总数时多余请求服务端排队")
    parser.add_argument("--context", type=int, default=CONTEXT,
                        help="llama-server 总上下文（-c）；0 = 自动 = 槽数 × (max-tokens + 256)")
    parser.add_argument("--questions-per-batch", type=int, default=QUESTIONS_PER_BATCH,
                        help="一次对话批量生成的问题数；=1 时自动取消批量机制（单题模式）")
    parser.add_argument("--buffer", type=int, default=None,
                        help="问题缓冲队列容量（默认 = --questions-per-batch 一整批）")
    parser.add_argument("--port", type=int,
                        default=SERVER_PORT, help="llama-server 端口")
    parser.add_argument("--keep-server", action="store_true",
                        help="结束时保留 llama-server 进程（下次可复用）")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.parallel < 1:
        sys.exit("--parallel 至少为 1")
    if args.questions_per_batch < 1:
        sys.exit("--questions-per-batch 至少为 1（=1 自动进入单题模式）")
    if args.producers < 1:
        sys.exit("--producers 至少为 1")
    if args.buffer is None:
        # 批量模式默认一整批；单题模式给 2 倍线程总数深度，避免生产者过度节流
        args.buffer = ((args.producers + args.parallel) * 2
                       if args.questions_per_batch <= 1
                       else args.questions_per_batch)
    if args.buffer < 1:
        sys.exit("--buffer 至少为 1")

    if not os.path.exists(LLAMA_SERVER_EXE):
        sys.exit(f"找不到 llama-server: {LLAMA_SERVER_EXE}")
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"找不到模型: {MODEL_PATH}")

    topics = [t.strip() for t in args.topics.split(",")
              ] if args.topics else TOPICS

    for p in (args.output_txt, args.output_jsonl):
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)

    # 去重集合同时读当前输出与旧版产物（若存在），避免跨运行去重断链
    seen = load_seen(args.output_jsonl, LEGACY_JSONL)

    # 槽位：默认与线程 1:1（无服务端排队）；手动调小则超额请求排队抢槽
    if args.slots < 0:
        sys.exit("--slots 不能为负")
    slots = args.slots if args.slots > 0 else args.producers + args.parallel
    # -c 会被 llama.cpp 均摊给每个槽位：按"槽数 × (单请求最大 token + 提示余量)"预留，
    # 保证所有槽位同时打满时也不会截断回答
    ctx = args.context if args.context > 0 else slots * (args.max_tokens + 256)

    server = LlamaServer(args.port, slots, args.gpu_layers, ctx)
    try:
        server.start()
    except Exception as e:
        sys.exit(f"[error] {e}")

    made = 0
    start_time = time.time()

    q_buf: queue.Queue = queue.Queue(maxsize=args.buffer)
    results: queue.Queue = queue.Queue()          # 无界：写盘远快于回答
    stop_event = threading.Event()
    seen_lock = threading.Lock()                  # 生产者增、主线程对失败项删

    print(f"[info] 模型: {MODEL_PATH}")
    print(f"[info] 主题池: {topics}")
    print(f"[info] 流水线: {args.producers} 出题线程 + {args.parallel} 回答线程 + 主线程写盘，"
          f"缓冲 {args.buffer} 题（槽位 {slots}，总上下文 {ctx}）")
    print(f"[info] 每对话批量问题数: {args.questions_per_batch}")
    print(f"[info] 历史样本: {len(seen)} 条（用于去重，含旧产物）")

    producers = [
        threading.Thread(
            target=produce_questions,
            args=(server, args, topics, seen, seen_lock, q_buf, stop_event, i),
            name=f"question-producer-{i}", daemon=True)
        for i in range(args.producers)
    ]
    workers = [
        threading.Thread(
            target=answer_worker,
            args=(server, args, i, q_buf, results, stop_event),
            name=f"answer-worker-{i}", daemon=True)
        for i in range(args.parallel)
    ]

    try:
        with open(args.output_txt, "a", encoding="utf-8") as ftxt, \
                open(args.output_jsonl, "a", encoding="utf-8") as fjson:
            for t in producers + workers:
                t.start()

            # 主线程写手：按完成顺序取结果、逐条落盘，满额即停
            recent_times = deque()   # 近 15s 完成时刻，用于瞬时速率
            while made < args.count:
                try:
                    q, qn, topic, a = results.get(timeout=0.5)
                except queue.Empty:
                    alive = any(t.is_alive() for t in producers + workers)
                    if not alive:
                        break  # 生产已终止且无在途结果
                    continue
                if a is None:
                    with seen_lock:
                        seen.discard(qn)  # 回答失败，允许后续重新生成该问题
                    print(f"  [warn] 回答生成多次失败，跳过: {q[:50]}")
                    continue
                ftxt.write(to_chatml_line(q, a) + "\n")
                fjson.write(json.dumps({
                    "id": f"synthetic_{int(time.time())}_{made:04d}",
                    "topic": topic,
                    "question": q,
                    "answer": a,
                    "q_norm": qn,
                }, ensure_ascii=False) + "\n")
                ftxt.flush()
                fjson.flush()
                made += 1
                now = time.time()
                recent_times.append(now)
                while now - recent_times[0] > 15:
                    recent_times.popleft()
                elapsed = now - start_time
                rate = made / elapsed * 60.0        # 本次运行累计均速
                rate_recent = len(recent_times) / min(15.0, elapsed) * 60.0
                eta = (args.count - made) * elapsed / made
                print(
                    f"[{made}/{args.count}] 近期 {rate_recent:.0f} 条/分 | 均速 {rate:.1f} 条/分 | ETA {eta:.0f}s | "
                    f"Q: {q[:10]}...A: {a[:40]}...")
            stop_event.set()
    except KeyboardInterrupt:
        stop_event.set()
        print("\n[interrupt] 已保存已完成的样本，下次运行将继续")
    finally:
        stop_event.set()
        for t in producers + workers:
            t.join(timeout=15)  # 线程均为 daemon，超时后随主进程退出
        if not args.keep_server:
            server.close()
        else:
            print(f"[info] 保留 llama-server（端口 {args.port}），下次运行自动复用")

    if made < args.count:
        print(f"[warn] 仅新增 {made}/{args.count} 条（问题生产提前终止或回答失败过多）")
    elapsed = time.time() - start_time
    rate_txt = f"，均速 {made / elapsed * 60:.1f} 条/分" if made else ""
    print(f"[done] 新增 {made} 条，耗时 {elapsed:.1f}s{rate_txt}")
    print(f"       输出: {args.output_txt}")
    print(f"       jsonl: {args.output_jsonl}")


if __name__ == "__main__":
    main()
