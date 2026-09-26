# tokenizer_archive

已退役的分词器快照，当前没有任何代码引用，仅作复现旧 run 之用。

- `bbpe_tokenizer_7k_260715.json`、`bbpe_tokenizer_7k_260723_l.json` — xl 版本的早期/l 变体
- `bbpe_tokenizer_pure_smoke.json` — `train_tokenizer_custom.py` 的 smoke 产物
- `bpe_tokenizer_6k_0717.json`、`bpe_tokenizer_7k_260215.json` — 更早的 6k/7k 版本

现役 tokenizer：根目录 `bbpe_tokenizer_7k_260723_xl.json`；
旧 SFT 链（`continue_training.py` 等）仍引用根目录 `bpe_tokenizer_6k_0724_ChatML.json`。
