# 判断两句话是否只差一个字
# 先去除停用词
import jieba
import synonyms

# 判断不同的关键字是否相似
from pypinyin import lazy_pinyin


# 1.去除无意义的词和标点符号，但是像“日常”这种怎么处理？
def remove_somechars(s):
    s = s.strip()
    s = s.replace('，', '').replace('。', '').replace('？', '').replace('：', '').replace('请问', '').replace('如果', '').replace(
        '、', '')
    # 创建停用词列表
    filepath = 'stopwords.txt'
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    #
    s_segment = jieba.cut(s)
    s_remove_stopwords = ''
    for word in s_segment:
        if word not in stopwords:
            if word != '\t':
                s_remove_stopwords += word
    return s_remove_stopwords

# 拼音一样的，直接返回1，不需要往下判断
def is_same_pinyin(s1, s2):
    if lazy_pinyin(s1) == lazy_pinyin(s2):
        return 1
    return 2

# 2.两个句子中不同的词
def is_fanyi(s1, s2):
    s1_segment = jieba.cut(s1)
    s2_segment = jieba.cut(s2)
    diff_word_list = []
    for w1 in s1_segment:
        if w1 not in s2_segment:
            diff_word_list.append(w1)
    for w2 in s2_segment:
        if w2 not in s1_segment:
            diff_word_list.append(w2)
    return diff_word_list

# 3.两个句子只差一个字，且不是反义词则标记为1
def only_one_diff():
    pass

# 如果只是词调换了顺序，则一定返回1
def is_change_order(s1, s2):
    # 判断是否相互包含
    diff_word_list = []
    for w1 in s1:
        if w1 not in s2:
            diff_word_list.append(w1)
    for w2 in s2:
        if w2 not in s1:
            diff_word_list.append(w2)
    if len(diff_word_list) == 0:
        return 1

    # 换字和换位置同时都有，那只用检验拼音是否相互包含即可
    diff_pinyin_list = []
    s1_pinyin_list = lazy_pinyin(s1)
    s2_pinyin_list = lazy_pinyin(s2)
    for p1 in s1_pinyin_list:
        if p1 not in s2_pinyin_list:
            diff_pinyin_list.append(p1)
    for p2 in s2_pinyin_list:
        if p2 not in s1_pinyin_list:
            diff_word_list.append(p2)
    if len(diff_pinyin_list) == 0:
        return 1

    return 2


# 返回1表示修正为1，0表示修正为0，2表示不确定
def glue(s1, s2):
    if is_same_pinyin(s1, s2) == 1:
        return 1
    if is_change_order(s1, s2) == 1:
        return 1


    # 去除停用词后并再次尝试
    s1 = remove_somechars(s1)
    s2 = remove_somechars(s2)
    if is_same_pinyin(s1, s2) == 1:
        return 1
    if is_change_order(s1, s2) == 1:
        return 1
    return 2


if __name__ == '__main__':

    s1 = ['车祸死亡赔偿金谁有权分配获得']
    s2 = ['谁有权分配获得车祸死亡赔偿金']
    s3 = ['谁有权分配获得车火死亡赔偿金']
    print(glue(s1[0], s2[0]))
    print(glue(s1[0], s3[0]))






# def sim_sentence(sen1, sen2):
#     r = synonyms.compare(sen1, sen2, seg=True)
#     print(r)
#
# sen1 = "发生历史性变革"
# sen2 = "发生历史性变革"  # 1
# sen1 = "坐牢"
# sen2 = "蹲号子"  # 0.023
# sen1 = "自首"
# sen2 = "被抓"  # 0.198
# sen1 = "夫妻"
# sen2 = "恋人"  # 0.895 实际上完全不同
# sen1 = "死亡"
# sen2 = "流产"  # 0.192
# sen1 = "有"
# sen2 = "没有" # nan
# sen1 = "有"
# sen2 = "没有"
# sim_sentence(sen1, sen2)
