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
    unk_token = '[UNK]'
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
    tokenizer: Tokenizer = Tokenizer.from_file("bpe_tokenizer_6k_0717.json")
    text = "[START][PAD][UNK]# 一朵一果|奖项合集（上）\n**一朵一果十周年**\n**一朵一果十周年啦！****A**家长十年来你们最大的收获是什么？一朵一果十年来我们给孩子们提供了许多大型的平台，为湘乡市培养了众多的艺术人才。**果****B**家长那我的孩子来学习会有大量收获吗？一朵一果一朵一果的课程安排都是体系化的，每一堂课是环环相扣的，我们的比赛、实践也是和课堂内容挂钩的，所以小朋友的艺术学习会获得不错的收获。同时，我们提供的平台也可以给学生更多的实践机会，不断的送去更大的舞台锻炼。**果**\n**0****1**\n**20142014年一朵一果学员陈琳婧第三届“夏青杯”朗诵大赛湖南赛区一等奖的第三名 、蒲公英第十四届青少年优秀艺术新人选拔大赛全国总决赛银奖\n2014年一朵一果播音系学员李婕第三届“夏青杯”朗诵大赛湖南赛区一等奖的第一名蒲公英、第十四届少年优秀新人选拔大赛全国总决赛金奖\n**优秀艺术新人****Outstanding New Artist2014年蒲公英第十四届青少年优秀艺术新人选拔活动一朵一果学员所获奖状\n**全国星姐****The national star elder sister2014年一朵一果形象代言人刘昭君荣获全国星姐选举亚军\n2014年易可威获得“我是金话筒”决赛湖南省唯一一个金奖、魏锦天获得湖南省的铜奖、参赛最小学员毛涵宇获得“最具潜力奖”。\n**02**\n**20152015年一朵一果原创语言类情景剧《为爱回乡》、原创音诗画表演《我不是完美小孩》代表湘乡参加湘潭市第五届中小学生艺术展演获得湘潭市第五届中小学生艺术展演“一等奖”同时代表湘潭市教育系统参加湖南省第五届中小学生艺术展演分别获得湖南省中学组“一等奖”、小学组 “一等奖”\n**03**\n**20162016年度“筑梦杯”全国青少年儿童书画大赛一朵一果收获大满贯\n其中，谭雅文、宴菲荣获特别金奖\n李晴、周征和何睿哲、杨礼嘉、向思璐、陈达、谭俣喆、周宇文荣获金奖\n彭湃、杨鑫淼、洪熠宇、李奕萱、陈蓓暄、周湘怡、王小苗荣获银奖\n贺姝婕、欧阳一诺、陈瑾瑜、李承轩荣获铜奖\n2016年钢琴系李婕、朱嘉颖与世界钢琴大师郎朗同台演出，获郎朗101钢琴大赛湖南省一等奖\n**04**\n**20172017年江苏广电“未来金话筒”大赛中，一朵一果高考班学员何沐骏荣获全国一等奖，学员蔡心融、曾子彧荣获全国三等奖\n**夏青杯****Xia Qing cup2017年一朵一果播音系高考班周湘沛、李婕参加“夏青杯”朗诵大赛，获得全国二等奖\n**演讲大赛****Speech contest2017年全国第二届学生“学宪法，讲宪法”演讲大赛，一朵一果播音系学员易可葳荣获湖南省一等奖\n2017年一朵一果播音系学员袁静雯获得少年中国说——首届中小学生口语表达能力展演大赛全国二等奖\n2017年一朵一果共13位小选手参加IPA完美童模全国总决赛，共获得一个男生G组总冠军、三个季军，女生共获得七个金奖、两个银奖\n**完美童模****IPA2017年一朵一果共13位小选手参加IPA完美童模全国总决赛，一朵一果战队获得了代表组的总冠军\n**声乐演唱****vocal2017年湖南省“欢乐潇湘”声乐表演唱《梨花又开放》荣获湘潭市二等奖\n**艺术百佳**\n2017年一朵一果声乐高考班学员陈琳婧、陈子涵、孙艺馨、章宇、孔雯妮、李锦泓获得第十六届“艺术百佳”一、二等奖\n**美术系**李林哲、李知典在2017年湖南省少儿才艺大赛，均荣获书法组铜奖\n谭雅文在2017年“溢美童心”第七届全国青少年儿童书画大赛中荣获金奖\n**精彩继续**一朵一果在这十年间培育了许多优秀学生，我们下期继续哦\n**END[END][START]广州市黄埔区开元学校（Guangzhou Kaiyuan School），是一所公办十二年制学校，是广州市第二中学教育集团成员校，是广东省义务教育标准化学校、广州市首批中小学（中职）思政课新结构教学评范式研究试点实验学校、广州市中小学教育高质量发展实验学校、广州市中小学心理健康教育特色学校。 [1] [6] [14] [29] [33-34]\n学校创办于2018年；学校于2020年被确认为广东省义务教育标准化学校，高中部于2021年9月开办，学校西校区（熙元校区）于2024年9月开办。 [1] [6-8]\n截至2024年9月，学校东校区总占地面积约125亩，建筑面积约125000平方米，学校西校区总用地面积42120平方米，总建筑面积65731平方米。截至2024年10月，两校区现总计已开设120个教学班。 [3] [5] [10] [34][END]"
    encoding = tokenizer.encode(text)

    # 获取token字符串（调试用）
    tokens = encoding.tokens

    # 获取token IDs（必须传给decode）
    token_ids = encoding.ids
    print(token_ids)

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
    
