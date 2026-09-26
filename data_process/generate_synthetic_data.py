"""
合成对话数据生成脚本（data_process/generate_synthetic_data.py）

流程（每一条样本）：
1. 批量问题：一个对话中让模型基于随机主题生成一批问题（QUESTIONS_PER_BATCH 个，
   减少请求次数），随后按编号解析并逐条校验
2. 去重：规范化后与历史已生成问题比对，重复则丢弃（可跨运行去重）
3. 新对话回答：每个问题单独送入全新上下文生成回答（避免"接着编"）
4. 校验 + 输出：一行一个样本，样本内换行转义为字面 \\n：
   "<|im_start|>user\\n问题<|im_end|>\\n<|im_start|>assistant\\n回答<|im_end|>\\n"
5. 断点续跑：启动时从已有 jsonl 载入去重集合，中断后重跑只会补新样本

并发提速：启动单个 llama-server（-np/--parallel N 个槽位），
问题批量生成与逐个回答生成均用 ThreadPoolExecutor 并发提交请求，
一个模型实例同时服务 N 路独立对话，避免每样本重复加载模型。

用法：
    uv run python data_process/generate_synthetic_data.py --count 50 --parallel 8
    uv run python data_process/generate_synthetic_data.py --topics 历史,科学 --count 20 --temp 1.0
"""

import argparse
import concurrent.futures
import json
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
import urllib.request

# ========================= 可配置常量 =========================

# llama.cpp 二进制目录（含 llama-server.exe）
LLAMA_BIN_DIR = r"D:\llama-b10333-bin-win-cuda-13.3-x64"
LLAMA_SERVER_EXE = os.path.join(LLAMA_BIN_DIR, "llama-server.exe")

# 模型路径（GGUF）
MODEL_PATH = r"D:\Model\Qwen3.5-4B-UD-Q4_K_XL.gguf"

# 输出文件（ChatML 每行一条对话；jsonl 含元数据、用于去重和断点续跑）
OUTPUT_TXT = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "synthetic_sft.txt")
OUTPUT_JSONL = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "synthetic_sft.jsonl")
SERVER_LOG = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "llama_server.log")

# llama-server 配置
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 18080
PARALLEL = 6                 # -np/--parallel：并发槽位数（并发请求数）
CONTEXT = 6144               # 上下文长度
SERVER_READY_TIMEOUT = 120   # 等待服务就绪（秒）

# 生成参数
NUM_SAMPLES = 1000             # 本次要生成的新样本数（去重后净增）
MAX_N_TOKENS = 512           # 单次生成最大 token 数
GPU_LAYERS = 99              # 送入 GPU 的层数（-ngl）
TEMPERATURE = 0.7              # 采样温度
SEED = -1                    # 随机种子（-1 = 每次随机）
MAX_RETRIES = 3              # 单条样本失败重试次数
REQUEST_TIMEOUT = 300        # 单次 HTTP 请求超时（秒）
QUESTIONS_PER_BATCH = 20      # 一个对话批量生成的问题数

# 问题生成的主题池（--topics 可覆盖）
TOPICS = [
    "中国历史", "日常生活", "日常对话", "科学常识", "编程技术", "文学创作",
    "心理健康", "学习方法", "数学", "科技数码", "自我介绍，身份问答"
    "美食", "旅行", "职业规划", "传统文化", "人工智能",
]

# 系统提示词
SYSTEM_QUESTION_PROMPT = (
    "你是一个中文数据生成助手。请基于给定的主题，生成一批互不相同、答案简单、"
    "有讨论价值的简单问题。每行一个问题，用\"1.\" \"2.\"等数字编号开头。"
    "只输出编号问题列表，不要任何解释、前缀或后缀。"
)
SYSTEM_ANSWER_PROMPT = (
    "你是一个乐于助人的中文AI助手。请认真回答用户的问题，"
    "给出简要的回答，不超过500字。"
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


def load_seen(out_jsonl: str) -> set:
    """载入历史问题的规范化集合（断点续跑去重）。"""
    seen = set()
    if os.path.exists(out_jsonl):
        with open(out_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["q_norm"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return seen


# ========================= llama-server 管理 =========================

class LlamaServer:
    """管理 llama-server 进程：启动（或复用已有实例）、健康检查、请求、关闭。"""

    def __init__(self, port: int, parallel: int, gpu_layers: int):
        self.port = port
        self.parallel = parallel
        self.gpu_layers = gpu_layers
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
            "-c", str(CONTEXT),
            "-np", str(self.parallel),
            "--host", SERVER_HOST,
            "--port", str(self.port),
            "--spec_type", "draft-mtp",
            "--spec-draft-n-max", "1",
            # 服务级禁用 Qwen 思考模式，所有请求生效
            "--chat-template-kwargs", '{"enable_thinking": false}',
            # 禁用 prompt cache（cache-ram 默认 8GB）：每次请求都是全新对话，
            # 缓存无法复用，实测内存会随请求数线性增长到 8GB 上限
            "--cache-ram", "0",
        ]
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

    def chat(self, messages: list, seed: int, max_tokens: int) -> str:
        """OpenAI 兼容接口 /v1/chat/completions，返回 assistant 文本。"""
        body = json.dumps({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": TEMPERATURE,
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


# ========================= 生成步骤（并发 worker） =========================

def generate_question_batch(server: LlamaServer, topic: str, seed: int,
                            n_questions: int) -> list[str]:
    """步骤 1：一个对话批量生成 n_questions 个问题（重试，全部失败返回 []）。"""
    messages = [
        {"role": "system", "content": SYSTEM_QUESTION_PROMPT},
        {"role": "user",
         "content": f"请基于主题「{topic}」生成 {n_questions} 个不同的问题。"},
    ]
    for attempt in range(MAX_RETRIES):
        try:
            raw = sanitize(server.chat(messages, seed + attempt, MAX_N_TOKENS))
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


def generate_answer(server: LlamaServer, question: str, seed: int) -> str | None:
    """步骤 3：把问题送入全新对话生成回答（内部重试，全部失败返回 None）。"""
    messages = [
        {"role": "system", "content": SYSTEM_ANSWER_PROMPT},
        {"role": "user", "content": question},
    ]
    for attempt in range(MAX_RETRIES):
        try:
            a = sanitize(server.chat(messages, seed +
                         attempt + 1000, MAX_N_TOKENS))
        except Exception as e:
            print(f"  [warn] 回答生成失败({e})，重试 {attempt + 1}/{MAX_RETRIES}")
            continue
        if validate_answer(a, normalize(question)):
            return a
    return None


# ========================= 主流程 =========================

def main() -> None:
    parser = argparse.ArgumentParser(description="用 llama.cpp 并发生成合成 Q&A 数据")
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
                        default=PARALLEL, help="llama-server 并发槽位数")
    parser.add_argument("--questions-per-batch", type=int, default=QUESTIONS_PER_BATCH,
                        help="一次对话批量生成的问题数")
    parser.add_argument("--port", type=int,
                        default=SERVER_PORT, help="llama-server 端口")
    parser.add_argument("--keep-server", action="store_true",
                        help="结束时保留 llama-server 进程（下次可复用）")
    args = parser.parse_args()

    if not os.path.exists(LLAMA_SERVER_EXE):
        sys.exit(f"找不到 llama-server: {LLAMA_SERVER_EXE}")
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"找不到模型: {MODEL_PATH}")

    topics = [t.strip() for t in args.topics.split(",")
              ] if args.topics else TOPICS
    seen = load_seen(args.output_jsonl)

    server = LlamaServer(args.port, args.parallel, args.gpu_layers)
    try:
        server.start()
    except Exception as e:
        sys.exit(f"[error] {e}")

    made = 0
    tried = 0
    start_time = time.time()
    rng = random.Random(args.seed if args.seed >= 0 else None)

    print(f"[info] 模型: {MODEL_PATH}")
    print(f"[info] 主题池: {topics}")
    print(f"[info] 并发槽位: {args.parallel}")
    print(f"[info] 每对话批量问题数: {args.questions_per_batch}")
    print(f"[info] 历史样本: {len(seen)} 条（用于去重）")

    try:
        with open(args.output_txt, "a", encoding="utf-8") as ftxt, \
                open(args.output_jsonl, "a", encoding="utf-8") as fjson:
            while made < args.count:
                # ---- 阶段 A：并发生成问题（每对话一批，需求数按 2 倍预留） ----
                need = min(args.parallel * 2, max(1, (args.count - made) * 2))
                n_jobs = max(1, math.ceil(need / args.questions_per_batch))
                batch_topics = [rng.choice(topics) for _ in range(n_jobs)]
                batch_seeds = [rng.randint(0, 2**31 - 1)
                               for _ in range(n_jobs)]
                with concurrent.futures.ThreadPoolExecutor(
                        max_workers=args.parallel) as ex:
                    futs = [ex.submit(generate_question_batch, server, t, s,
                                      args.questions_per_batch)
                            for t, s in zip(batch_topics, batch_seeds)]
                    q_lists = [f.result() for f in futs]
                # 展平并保留每问题所属主题
                questions = [(q, t) for qs, t in zip(
                    q_lists, batch_topics) for q in qs]
                tried += len(questions)
                if not questions:
                    print("  [warn] 本批问题生成全部失败，重试")
                    continue

                # ---- 阶段 B：去重（含批内去重） ----
                valid = []
                for q, t in questions:
                    qn = normalize(q)
                    if qn in seen:
                        print(f"  [dup] 问题已存在，跳过: {q[:50]}")
                        continue
                    seen.add(qn)
                    valid.append((q, qn, t))
                if not valid:
                    print("  [warn] 本批无有效新问题，重试")
                    continue

                # ---- 阶段 C：并发生成回答（每问题一个全新对话） ----
                with concurrent.futures.ThreadPoolExecutor(
                        max_workers=args.parallel) as ex:
                    futs = [ex.submit(generate_answer, server, q,
                                      rng.randint(0, 2**31 - 1)) for q, _, _ in valid]
                    answers = [f.result() for f in futs]

                # ---- 阶段 D：写入（最多写到剩余配额，避免一批超额） ----
                for (q, qn, t), a in zip(valid, answers):
                    if made >= args.count:
                        break
                    if a is None:
                        seen.discard(qn)  # 回答失败，允许后续重新生成该问题
                        print(f"  [warn] 回答生成多次失败，跳过: {q[:50]}")
                        continue
                    ftxt.write(to_chatml_line(q, a) + "\n")
                    fjson.write(json.dumps({
                        "id": f"synthetic_{int(time.time())}_{made:04d}",
                        "topic": t,
                        "question": q,
                        "answer": a,
                        "q_norm": qn,
                    }, ensure_ascii=False) + "\n")
                    ftxt.flush()
                    fjson.flush()
                    made += 1
                    elapsed = time.time() - start_time
                    print(
                        f"[{made}/{args.count}] Q: {q[:10]}...A: {a[:40]}...({elapsed:.0f}s)")
    except KeyboardInterrupt:
        print("\n[interrupt] 已保存已完成的样本，下次运行将继续")
    finally:
        if not args.keep_server:
            server.close()
        else:
            print(f"[info] 保留 llama-server（端口 {args.port}），下次运行自动复用")

    print(
        f"[done] 新增 {made} 条，累计尝试 {tried} 次，耗时 {time.time() - start_time:.1f}s")
    print(f"       输出: {args.output_txt}")
    print(f"       jsonl: {args.output_jsonl}")


if __name__ == "__main__":
    main()
