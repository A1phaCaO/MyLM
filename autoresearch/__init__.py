def __getattr__(name):
    import importlib
    module_map = {
        "create_attention": ".benchmark",
        "benchmark_attention": ".benchmark",
        "verify_correctness": ".benchmark",
        "compute_score": ".benchmark",
        "DEFAULT_BENCHMARK_CONFIG": ".benchmark",
        "get_device": ".benchmark",
        "EvolutionEnv": ".evolution_env",
    }
    if name in module_map:
        mod = importlib.import_module(module_map[name], __package__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "create_attention",
    "benchmark_attention",
    "verify_correctness",
    "compute_score",
    "DEFAULT_BENCHMARK_CONFIG",
    "get_device",
    "EvolutionEnv",
]
