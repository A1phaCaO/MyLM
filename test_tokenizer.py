from tokenizers import Tokenizer
import tokenizers
import tokenizers.decoders
from typing import Tuple, Dict


def count_unknown_words(text: str, tokenizer: Tokenizer) -> Tuple[Dict[str, int], int]:
    """对给定文本进行编码，并返回未知词及其计数。
    
    参数:
        text (str): 需要分析的文本。
        tokenizer (Tokenizer): 用于编码文本的tokenizer。
    
    返回:
        Tuple[Dict[str, int], int]: 包含未知词字典和未知词总数。
    """
    # 使用tokenizer的预分词器获取原始单词列表
    pre_tokenized = list(text)
    words = [word for word in pre_tokenized]
    
    # 获取未知词的ID
    unk_token = '<|unk|>'
    unk_id = tokenizer.token_to_id(unk_token)
    if unk_id is None:
        # 如果tokenizer没有unk_token，则尝试通过编码获取
        encoded_unk = tokenizer.encode(unk_token)
        if len(encoded_unk.ids) > 0:
            unk_id = encoded_unk.ids[0]
        else:
            # 如果无法获取unk_id，则无法检测未知词
            return {}, 0
    
    # 统计未知词
    unknown_words = {}
    total_unk = 0
    
    # 检查每个单词是否被识别
    for word in words:
        # 编码单个单词
        encoded_word = tokenizer.encode(word)
        
        # 如果编码结果包含未知标记或整个单词被标记为未知
        if any(token_id == unk_id for token_id in encoded_word.ids):
            total_unk += 1
            unknown_words[word] = unknown_words.get(word, 0) + 1
    
    return unknown_words, total_unk




if __name__ == "__main__":
    tokenizer: Tokenizer = Tokenizer.from_file("bpe_tokenizer_6k_0724_ChatML.json")
    text = r"<|im_start|>system\n你是一个乐于助人的AI助手，用中文进行对话。<|im_end|>\n<|im_start|>user\n你好，今天天气怎么样？<|im_end|>\n<|im_start|>assistant\n你好！我无法获取实时天气信息，建议你查看当地的天气预报应用或网站哦。<|im_end|>\n<|im_start|>user\n你能帮我写一首关于秋天的诗吗？<|im_end|>\n<|im_start|>assistant\n当然可以，这是一首为你写的秋日小诗：\n\n秋风轻拂叶飘黄，\n霜染层林映晚阳。\n稻浪翻金农事乐，\n雁行南去字成行。\n\n希望你喜欢！<|im_end|>"
    encoding = tokenizer.encode(text)

    # 获取token字符串（调试用）
    tokens = encoding.tokens

    # 获取token IDs（必须传给decode）
    token_ids = encoding.ids
    print(token_ids)
    word2idx = tokenizer.get_vocab()
    line_ = []
    print(encoding.tokens)
    for i in encoding.tokens:
        try:
            line_.append(word2idx[i])
        except KeyError:
            line_.append(word2idx["<|unk|>"])
    print(line_)
    decoded_text = tokenizer.decode(token_ids, skip_special_tokens=False)
    tokenizers.decoders.BPEDecoder(decoded_text)
    print("解码结果:", decoded_text)
    print(
        f"token数: {len(token_ids)}, 字符数: {len(text)}, 压缩率: {len(token_ids) / len(text)}"
    )

    # 使用 count_unknown_words 函数
    unknown_words, unknown_count = count_unknown_words(text, tokenizer)

    print(f"未知词: {unknown_words}")
    print(f"未知词数: {unknown_count}")
    
