import torch
import torch.nn.functional as F
import json
from tokenizers import Tokenizer
import models as m

# 路径常量
MODEL_DIR = r"model\model_state_0724.pth"
TOKENIZER_DIR = r"bbpe_tokenizer_7k_260723_xl.json"
CONFIG_DIR = r"model\config_0724.json"
TOP_K = 10


def load_model_and_tokenizer():
    """加载 config、tokenizer、模型权重，返回 (embeddings, tokenizer, special_ids)"""
    # 1. 读 config
    with open(CONFIG_DIR, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 2. 加载 tokenizer
    tokenizer = Tokenizer.from_file(TOKENIZER_DIR)

    # 3. 构造 MyLMArgs（vocab_size 以 tokenizer 为准；dropout=0；其余字段对齐 config）
    args = m.MyLMArgs(
        d_model=config["d_model"],
        d_inner=config["d_inner"],
        n_layers=config["n_layers"],
        use_moe=config["use_moe"],
        n_experts=config["n_experts"],
        n_heads=config["n_heads"],
        d_head=config["d_head"],
        vocab_size=tokenizer.get_vocab_size(),
        seq_max_len=config["seq_max_len"],
        conv_bias=False,
        ffn_bias=config["ffn_bias"],
        attn_bias=config["attn_bias"],
        dropout=0,
    )

    # 4. 实例化模型（CPU 即可，仅需嵌入矩阵）
    model = m.MyLM(args).to("cpu")

    # 5. 加载权重（参考 run_model_for_state.py 的容错逻辑）
    raw = torch.load(MODEL_DIR, map_location="cpu", weights_only=False)
    # 兼容 checkpoint dict 与裸 state_dict
    if isinstance(raw, dict) and "model_state_dict" in raw and not any(
        k.endswith(".weight") or k.endswith(".bias") for k in raw.keys()
    ):
        print("检测到 checkpoint dict，提取 model_state_dict")
        state_dict = raw["model_state_dict"]
    else:
        state_dict = raw

    # 兼容多种前缀：DataParallel 的 module. 与 torch.compile 的 _orig_mod.
    PREFIXES = ("module.", "_orig_mod.")
    if any(k.startswith(PREFIXES) for k in state_dict.keys()):
        stripped = {}
        for k, v in state_dict.items():
            for p in PREFIXES:
                if k.startswith(p):
                    k = k[len(p):]
                    break
            stripped[k] = v
        state_dict = stripped
        print(f"已剥离前缀 {PREFIXES}")

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"严格加载失败：{str(e)[:80]}...")
        miss, unexpect = model.load_state_dict(state_dict, strict=False)
        print(f"已使用非严格加载：缺失 {len(miss)} 个，未匹配 {len(unexpect)} 个")

    # 6. 提取嵌入矩阵
    embeddings = model.token_embedding.weight.detach().cpu()
    print(f"嵌入矩阵形状：{tuple(embeddings.shape)}")

    # 7. 收集特殊 token id（key 以 < 开头、以 > 结尾）
    vocab = tokenizer.get_vocab()
    special_ids = {
        tid for tok, tid in vocab.items() if tok.startswith("<") and tok.endswith(">")
    }
    print(f"已标记 {len(special_ids)} 个特殊 token 将被过滤")

    return embeddings, tokenizer, special_ids


def find_similar(tokenizer, embeddings, special_ids, query_word, top_k=TOP_K):
    """对输入词进行 BBPE 编码、均值池化、计算余弦相似度、返回 top_k 结果"""
    # 1. 编码（tokenizers.encode 不支持 skip_special_tokens，需手动过滤）
    enc = tokenizer.encode(query_word)
    # 过滤掉可能混入的特殊 token（通常 BBPE 不会自动添加，除非输入字面包含）
    pairs = [(tid, tok) for tid, tok in zip(enc.ids, enc.tokens) if tid not in special_ids]
    token_ids = [p[0] for p in pairs]
    sub_tokens = [p[1] for p in pairs]

    if not token_ids:
        print("  [词表无法编码该输入，请尝试其他字符]")
        return

    # 2. 打印子 token 切分（用 decode 还原为可读文本，BBPE 字节形式如 èĭ¹æŀľ 不直观）
    sub_tokens_display = []
    for tid, raw_tok in zip(token_ids, sub_tokens):
        decoded = tokenizer.decode([tid])
        # 单字节 token decode 可能为空（多字节字符的片段），回退到字节形式
        sub_tokens_display.append(decoded if decoded else raw_tok)
    print(f"  子 token 切分 ({len(token_ids)} 个): {sub_tokens_display}")

    # 3. 取子 token 嵌入并均值池化
    idx_tensor = torch.tensor(token_ids, dtype=torch.long)
    sub_vecs = embeddings[idx_tensor]  # (n_sub, d_model)
    query_vec = sub_vecs.mean(dim=0)   # (d_model,)

    # 4. 余弦相似度
    emb_norm = F.normalize(embeddings, dim=1)        # (V, d)
    q_norm = F.normalize(query_vec, dim=0)           # (d,)
    sims = emb_norm @ q_norm                          # (V,)

    # 5. 构造屏蔽：特殊 token + 查询自身的 token id
    exclude = set(special_ids) | set(token_ids)
    mask = torch.ones_like(sims, dtype=torch.bool)
    for tid in exclude:
        if 0 <= tid < sims.shape[0]:
            mask[tid] = False
    sims = sims.masked_fill(~mask, float("-inf"))

    # 6. top_k
    k = min(top_k, int(mask.sum().item()))
    if k == 0:
        print("  [无可用候选 token]")
        return
    top_scores, top_ids = torch.topk(sims, k=k)

    # 7. 输出
    print(f"  Top-{k} 近邻：")
    for rank, (score, tid) in enumerate(zip(top_scores.tolist(), top_ids.tolist()), 1):
        text = tokenizer.decode([tid], skip_special_tokens=False)
        # 显示用：把换行/空白替换为可见字符，避免排版混乱
        text_show = repr(text) if (text == "" or any(c in text for c in "\n\r\t")) else text
        print(f"    {rank:>2}. {text_show:<10}  score={score:.4f}  id={tid}")


def main():
    print("=" * 60)
    print("词向量近邻检索（基于 token_embedding，余弦相似度）")
    print("=" * 60)
    embeddings, tokenizer, special_ids = load_model_and_tokenizer()
    print("-" * 60)
    print("输入一个词查找向量空间中的相近词。")
    print("命令：quit 退出 | K=20 修改 top_k（当前 K=10）")
    print("-" * 60)

    top_k = TOP_K
    while True:
        try:
            word = input("Word>>").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not word:
            continue
        low = word.lower()
        if low in ("quit", "exit", "q"):
            print("再见。")
            break
        if low.startswith("k="):
            try:
                top_k = max(1, int(low[2:]))
                print(f"  top_k 已设置为 {top_k}")
            except ValueError:
                print("  [格式应为 K=数字，例如 K=20]")
            continue

        find_similar(tokenizer, embeddings, special_ids, word, top_k=top_k)


if __name__ == "__main__":
    main()
