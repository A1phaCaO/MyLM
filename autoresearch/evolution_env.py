import json
import os
import time
import torch
from typing import Any, Dict, List, Optional, Tuple

from .benchmark import (
    create_attention,
    get_device,
    benchmark_attention,
    verify_correctness,
    compute_score,
    DEFAULT_BENCHMARK_CONFIG,
)


class EvolutionEnv:
    def __init__(
        self,
        history_path: str = "autoresearch/leaderboard.json",
        device: Optional[torch.device] = None,
        benchmark_config: Optional[Dict[str, Any]] = None,
    ):
        self.history_path = os.path.abspath(history_path)
        self.device = device or get_device()
        self.benchmark_config = dict(DEFAULT_BENCHMARK_CONFIG)
        if benchmark_config is not None:
            self.benchmark_config.update(benchmark_config)
        self.history: List[Dict[str, Any]] = []
        self._load_history()

    # ── persistence ──────────────────────────────────────────────

    def _load_history(self):
        if os.path.exists(self.history_path):
            with open(self.history_path, "r") as f:
                self.history = json.load(f)

    def _save_history(self):
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        with open(self.history_path, "w") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    # ── properties ───────────────────────────────────────────────

    @property
    def best_score(self) -> float:
        return max((e["score"] for e in self.history), default=0.0)

    @property
    def best_entry(self) -> Optional[Dict[str, Any]]:
        return max(self.history, key=lambda e: e["score"]) if self.history else None

    @property
    def current_generation(self) -> int:
        return max((e["generation"] for e in self.history), default=0)

    @property
    def total_attempts(self) -> int:
        return len(self.history)

    # ── evaluation ───────────────────────────────────────────────

    def evaluate(
        self,
        use_gate: bool = False,
    ) -> Tuple[float, float, bool, str]:
        cfg = self.benchmark_config
        attn = create_attention(
            d_model=cfg["d_model"],
            d_head=cfg["d_head"],
            n_heads=cfg.get("n_heads"),
            seq_max_len=cfg["seq_len"],
            use_gate=use_gate,
            dropout=cfg["dropout"],
        )

        ok, msg = verify_correctness(
            attn,
            device=self.device,
        )

        if not ok:
            return 0.0, 0.0, False, msg

        tput = benchmark_attention(
            attn,
            batch_size=cfg["batch_size"],
            seq_len=cfg["seq_len"],
            device=self.device,
            n_warmup=cfg["n_warmup"],
            n_iters=cfg["n_iters"],
        )
        score = compute_score(tput, ok)
        return tput, score, True, msg

    # ── submission ───────────────────────────────────────────────

    def submit(
        self,
        agent_name: str,
        score: float,
        throughput: float,
        is_correct: bool,
        code_description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "agent_name": agent_name,
            "generation": self.current_generation + 1,
            "score": score,
            "throughput": throughput,
            "is_correct": is_correct,
            "code_description": code_description,
            "parameters": parameters or {},
            "benchmark_config": dict(self.benchmark_config),
        }
        self.history.append(entry)
        self._save_history()
        return entry

    # ── introspection ────────────────────────────────────────────

    def leaderboard(self, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        sorted_ = sorted(self.history, key=lambda e: e["score"], reverse=True)
        return sorted_[:top_k] if top_k is not None else sorted_

    def status(self) -> Dict[str, Any]:
        return {
            "total_attempts": self.total_attempts,
            "current_generation": self.current_generation,
            "best_score": self.best_score,
            "best_agent": self.best_entry["agent_name"] if self.best_entry else None,
            "correct_attempts": sum(1 for e in self.history if e["is_correct"]),
            "device": str(self.device),
        }


if __name__ == "__main__":
    env = EvolutionEnv()
    print("=== EvolutionEnv Status ===")
    for k, v in env.status().items():
        print(f"  {k}: {v}")
    print()
    print("Running evaluation (may take a while)...")
    tput, score, ok, msg = env.evaluate()
    print(f"  throughput: {tput:,.0f} tokens/sec")
    print(f"  score:      {score:,.0f}")
    print(f"  correct:    {ok}")
    print(f"  message:    {msg}")
