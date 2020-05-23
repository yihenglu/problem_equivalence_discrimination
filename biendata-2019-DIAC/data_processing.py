#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import pandas as pd
import re
from pandas import DataFrame
from pypinyin import pinyin, lazy_pinyin

from rule import glue

'''
以下几个函数为一个整体，去除数据中的杂乱部分，预处理掉类似这种内容：
1. ';}Ie_+='';Kc_(sy_,Ie_);}}function ht_(){var Sm_=aJ_();for (sy_=0; sy_< wF_.length;aT_++){var nv_=oX_(wF_[aT_],'_');Ie_+='';}Ie_+='';Kc_(sy_,Ie_);}else{var wF_= oX_(nK_[sy_],',');var Ie_='';for (aT_=0; aT_< wF_.length;aT_++){Ie_+=Mg_(wF_[
2. 緈冨_吶庅羙 2017/02/04 18:43:53 发表在 19楼
'''
def convert_to_submit(result_csv='output_robert_large_epoch9_lr1_ml48_bs16/test_results.tsv',
                      result_file='robert_large_epoch9_lr1_ml48_bs16_all_cols.csv',
                      submit_file='robert_large_epoch9_lr1_ml48_bs16.csv',
                      rule=1, is_test=False):
    '''
    将bert输出文件转化为比赛所需的提交格式，is_tset区分测试与验证
    :param result_csv:bert模型的logits文件
    :param result_file:含有测试文件的所有信息+预测的标签
    :param submit_file:result_file中删除某些列
    :param rule:规则1：正常 规则2：调整阈值 规则3：其它场外信息（如18个空格）
    :param is_test:是否是无标签的测试文件
    :return:
    '''
    import os
    print('find file: {} :{}'.format(result_csv, os.path.isfile(result_csv)))
    result_df = pd.read_csv(result_csv, encoding='utf-8', sep='\t', header=None)
    print(result_df.shape)

    if rule == 1:
        label_index = list(result_df.idxmax(axis=1))
    elif rule == 2:
        # 调整 0 1 类别的划分阈值
        label_index = []
        change_index_list = []  # 被调整的样本的索引
        positive_num = 0
        for index,row in result_df.iterrows():
            if row[1] > 0.989:  # 之前写的0.55，效果降低了0.0002，0.45还没有试过
                label_index.append(1)
                positive_num += 1
            else:
                label_index.append(0)
                if row[1] > 0.5:
                    change_index_list.append(index)
    # print(label_index)
    print('positive:{} negitive:{}'.format(positive_num, result_df.shape[0]-positive_num))

    test_data_df = pd.read_csv('data/dev_set.csv', encoding='utf-8', sep='\t', header=0)

    # 没有表名就添加表名
    if is_test:
        test_data_df.columns = ['id', 'question1', 'question2']
    else:
        test_data_df.columns = ['id', 'flag', 'title', 'context']

    test_data_df['pre_label'] = label_index
    test_data_df.to_csv(result_file, encoding='utf-8', sep='\t', index=None)

    # 输出被调整的样本
    for index, item in test_data_df.iterrows():
        if index in change_index_list:
            print('--------------------------------------------------------')
            print(item['id'], item['question1'], item['question2'], item['pre_label'])
    print('共调整了{}/{}条数据'.format(len(change_index_list), test_data_df.shape[0]))


    submit_df = test_data_df[['id', 'pre_label']]
    submit_df.to_csv(submit_file, encoding='utf-8', sep='\t', index=None, header=None)
    return

convert_to_submit(result_csv='robert_wwm_large_ext_f5k_epoch1_lr1e5_ml64_bs12_logits.csv',
                  result_file='robert_wwm_large_ext_f5k_epoch1_lr1e5_ml64_bs12_all_cols_rule2_989.csv',
                  submit_file='robert_wwm_large_ext_f5k_epoch1_lr1e5_ml64_bs12_rule2_989.csv',
                  rule=2, is_test=True)
# exit()


def gold_rule(id_question1_question2_prelabel_file, id_question1_question2_prelabel_corlabel_file, id_corlabel_submit_file):
    '''
    测试文件的所有信息：id	question1	question2	pre_label （\t）
    :param id_question1_question2_prelabel_file:包含测试文件的所有信息
    :param id_question1_question2_prelabel_corlabel_file:包含测试文件的所有信息 多了一列，表示纠错后的标签
    :param id_corlabel_submit_file:提交的文件形式，cor表示correct
    :return:
    '''
    id_question1_question2_prelabel_df = pd.read_csv(id_question1_question2_prelabel_file, encoding='utf-8', sep='\t', header=0)
    correct_flag_list = []  # 存放根据规则纠正以后的标签
    different_num = 0
    # 根据规则处理每一行
    for index, row in id_question1_question2_prelabel_df.iterrows():
        question1_str = str(row['question1']).replace('，','').replace('。','').replace('？','').replace('：','').replace('请问','').replace('如果','').replace('、','')  # 需要改！！
        question2_str = str(row['question2']).replace('，','').replace('。','').replace('？','').replace('：','').replace('请问','').replace('如果','').replace('、','')
        # if lazy_pinyin(question1_str) == lazy_pinyin(question2_str):
        if glue(question1_str, question2_str) == 1:
            print('规则认定为1的项')
            print(str(row['id']) + '\n' + str(row['question1']) + '\n' + str(row['question2']) + '\n')
            correct_flag_list.append(1)
            if row['pre_label'] != 1:
                different_num += 1
                print('---------将0改为1的项-------------')
                print(str(row['id']) + '\n' + str(row['question1']) + '\n' + str(row['question2']) + '\n')
        else:  # 按照原来的预测标签
            correct_flag_list.append(row['pre_label'])
    print('has {}/{}={} deifferent flags'.format(different_num, id_question1_question2_prelabel_df.shape[0],
                                                 different_num / id_question1_question2_prelabel_df.shape[0]))

    id_question1_question2_prelabel_df['cor_label'] = correct_flag_list
    id_question1_question2_prelabel_df.to_csv(id_question1_question2_prelabel_corlabel_file, encoding='utf-8', sep='\t', index=None)
    id_question1_question2_prelabel_df[['id', 'cor_label']].to_csv(id_corlabel_submit_file, encoding='utf-8', sep='\t', index=None, header=None)
    return

id_question1_question2_prelabel_file = 'robert_wwm_large_ext_f5k_epoch1_lr1e5_ml64_bs12_all_cols_rule2_989.csv'
id_question1_question2_prelabel_corlabel_file = 'robert_wwm_large_ext_f5k_epoch1_lr1e5_ml64_bs12_all_cols_rule2_989_corlabel.csv'
id_corlabel_submit_file = 'robert_wwm_large_ext_f5k_epoch1_lr1e5_ml64_bs12_rule2_989_corlabel_submit_file.csv'
gold_rule(id_question1_question2_prelabel_file, id_question1_question2_prelabel_corlabel_file, id_corlabel_submit_file)
# exit()


def gold_rule2(id_question1_question2_prelabel_file, id_question1_question2_prelabel_corlabel_file, id_corlabel_submit_file):
    '''
    测试文件的所有信息：id	question1	question2	pre_label （\t）
    :param id_question1_question2_prelabel_file:包含测试文件的所有信息
    :param id_question1_question2_prelabel_corlabel_file:包含测试文件的所有信息 多了一列，表示纠错后的标签
    :param id_corlabel_submit_file:提交的文件形式，cor表示correct
    :return:
    '''
    id_question1_question2_prelabel_df = pd.read_csv(id_question1_question2_prelabel_file, encoding='utf-8', sep='\t', header=0)
    correct_flag_list = []  # 存放根据规则纠正以后的标签
    different_num = 0
    # 根据规则处理每一行
    for index, row in id_question1_question2_prelabel_df.iterrows():
        question1_str = str(row['question1']).replace('，','').replace('。','').replace('？','').replace('：','').replace('请问','').replace('如果','').replace('、','')  # 需要改！！
        question2_str = str(row['question2']).replace('，','').replace('。','').replace('？','').replace('：','').replace('请问','').replace('如果','').replace('、','')
        # if lazy_pinyin(question1_str) == lazy_pinyin(question2_str):
        if glue(question1_str, question2_str) == 1:
            print('规则认定为1的项')
            print(str(row['id']) + '\n' + str(row['question1']) + '\n' + str(row['question2']) + '\n')
            correct_flag_list.append(1)
            if row['pre_label'] != 1:
                different_num += 1
                print('---------将0改为1的项-------------')
                print(str(row['id']) + '\n' + str(row['question1']) + '\n' + str(row['question2']) + '\n')
        else:  # 按照原来的预测标签
            correct_flag_list.append(row['pre_label'])
    print('has {}/{}={} deifferent flags'.format(different_num, id_question1_question2_prelabel_df.shape[0],
                                                 different_num / id_question1_question2_prelabel_df.shape[0]))

    id_question1_question2_prelabel_df['cor_label'] = correct_flag_list
    id_question1_question2_prelabel_df.to_csv(id_question1_question2_prelabel_corlabel_file, encoding='utf-8', sep='\t', index=None)
    id_question1_question2_prelabel_df[['id', 'cor_label']].to_csv(id_corlabel_submit_file, encoding='utf-8', sep='\t', index=None, header=None)
    return

# id_question1_question2_prelabel_file = 'robert_wwm_large_epoch5_lr1_ml64_bs12_all_cols_rule2_45.csv'
# id_question1_question2_prelabel_corlabel_file = 'robert_wwm_large_epoch5_lr1_ml64_bs12_all_cols_rule2_45_corlabel.csv'
# id_corlabel_submit_file = 'robert_wwm_large_epoch5_lr1_ml64_bs12_rule2_45_corlabel_submit_file.csv'
# gold_rule(id_question1_question2_prelabel_file, id_question1_question2_prelabel_corlabel_file, id_corlabel_submit_file)
# exit()


def compare_file(file1, file2):
    '''
    对比两个文件内容是否一样
    :param file1:
    :param file2:
    :return:
    '''
    temp_file = open('data/different.csv', 'w+', encoding='utf-8')
    f1_df = pd.read_csv(file1, encoding='utf-8', sep=',')
    f2_df = pd.read_csv(file2, encoding='utf-8', sep=',')
    different_num = 0
    for row1, row2 in zip(f1_df.itertuples(), f2_df.itertuples()):
        if getattr(row1, "correct_flag") != getattr(row2, "correct_flag"):
            different_num += 1
            # print(getattr(row1, "pre_label"), getattr(row2, "pre_label"))
            # print(getattr(row1, 'title'), getattr(row1, 'context'))
            temp_file.write(str(getattr(row1, "id")) + "," +
                            str(getattr(row1, "pre_label")) + "," +
                            str(getattr(row1, 'title')) +
                            str(getattr(row1, 'context')) +
                            str(getattr(row2, "pre_label"))+ '\n')
            print(row1)

    temp_file.close()

    print('has {}/{}={} deifferent flags'.format(different_num, f1_df.shape[0], different_num/f1_df.shape[0]))

# file1 = 'new_data_epoch9_glue1_corrected_all_cols.csv'
# file2 = 'corrected_all_cols.csv'
# compare_file(file1, file2)


def compare_result_file(result_file1, result_file2, test_file, result_file=None):
    '''
    把3个文件合并到一块，输入不同的项
    '''
    temp_file = open('biendata-2019-DIAC/different.csv', 'w+', encoding='utf-8')
    f1_df = pd.read_csv(result_file1, encoding='utf-8', sep='\t', header=None, index_col=0)
    f1_df.columns= ['pre_label']
    f2_df = pd.read_csv(result_file2, encoding='utf-8', sep='\t', header=None, index_col=0)
    f2_df.columns = ['pre_label_logits']
    test_df = pd.read_csv(test_file, encoding='utf-8', sep='\t')

    result_file = open('biendata-2019-DIAC/different_result.csv', 'w+', encoding='utf-8')

    different_num = 0
    index_num = 0
    for row1, row2, row3 in zip(f1_df.itertuples(), f2_df.itertuples(), test_df.itertuples()):
        sample_result = str(getattr(row1, "pre_label"))  # 只要两个文件结果不一样，就算成1

        if getattr(row1, "pre_label") != getattr(row2, "pre_label_logits"):
            different_num += 1
            # print(getattr(row1, "pre_label"), getattr(row2, "pre_label"))
            # print(getattr(row1, 'title'), getattr(row1, 'context'))
            temp_file.write(str(getattr(row3, "qid")) + "\t" +
                            str(getattr(row1, "pre_label")) + "\t" +
                            str(getattr(row2, 'pre_label_logits')) + '\t' +
                            str(getattr(row3, 'question1')) + '\t' +
                            str(getattr(row3, "question2"))+ '\n')

            sample_result = '1'
            print(row1)
        result_file.write(str(index_num) + '\t' + sample_result + '\n')
        index_num += 1

    temp_file.close()
    result_file.close()

    print('has {}/{}={} deifferent flags'.format(different_num, f1_df.shape[0], different_num/f1_df.shape[0]))

# file1 = 'biendata-2019-DIAC/bert_base_chinese_epoch5_lr1_ml64_bs32.csv'
# file2 = 'biendata-2019-DIAC/bert_base_chinese_epoch5_lr1e5_ml128_bs16_100000_logits_vote.csv'
# test_file = 'data/dev_set.csv'
# compare_result_file(file1, file2, test_file)
# exit()


def result_logits_merge(input_file1, input_file2, input_file3):
    '''合并3（多）个文件的标签'''
    input1_df = pd.read_csv(input_file1, encoding='utf-8', sep='\t', header=None)  # 注意不是/t
    input2_df = pd.read_csv(input_file2, encoding='utf-8', sep='\t', header=None)
    input3_df = pd.read_csv(input_file3, encoding='utf-8', sep='\t', header=None)
    label_logits0_merge = []
    label_logits1_merge = []
    for i1, i2, i3 in zip(input1_df.values, input2_df.values, input3_df.values):
        label_logits0_merge.append(i1[0] + i2[0] + i3[0])
        label_logits1_merge.append(i1[1] + i2[1] + i3[1])

    c = {"label_logits0_merge": label_logits0_merge,
         "label_logits1_merge": label_logits1_merge}  # 将列表a，b转换成字典
    result_logits_merge_df = DataFrame(c)  # 将字典转换成为数据框
    result_logits_merge_df.to_csv("final_merge_logits_test_results.tsv", encoding='utf-8', header=None, index=None, sep='\t')

    convert_to_submit(result_csv='final_merge_logits_test_results.tsv',
                      result_file='final_large_robert_epoch399_glue1_all_cols.csv', submit_file='final_large_robert_epoch399_glue1.csv', rule=1, is_test=True)

# result_logits_merge('output_robert_wwm_large_epoch6_lr1_ml128_bs4/test_results.tsv',
#                     'bert_output_robert_final/test_results.tsv',
#                     'bert_output_robert_89/test_results.tsv')
# exit()


def merge_train_test_file(train_file, has_label_test_file, new_train_file):
    '''
    合并原始训练集与带有预测标签的测试集
    :param train_file:
    :param has_label_test_file:
    :return:
    '''
    train_df = pd.read_csv(train_file, encoding='utf-8', sep=',')
    has_label_test = pd.read_csv(has_label_test_file, encoding='utf-8', sep=',')

    # 删除pre_label
    has_label_test.drop(labels=['pre_label'], axis=1, inplace=True)

    # 调整列的顺序
    flag = has_label_test['flag']
    has_label_test.drop(labels=['flag'], axis=1, inplace=True)
    has_label_test.insert(1, 'flag', flag)
    print(has_label_test.head())

    new_train_df = pd.concat([train_df, has_label_test], axis=0)

    from sklearn.utils import shuffle
    new_train_df = shuffle(new_train_df)

    new_train_df.to_csv(new_train_file, encoding='utf-8', sep=',', index=None)
    return

# train_file = 'data/train.csv'
# has_label_test_file = 'epoch9_glue1_corrected_all_cols.csv'
# new_train_file = 'new_train_file.csv'
# merge_train_test_file(train_file, has_label_test_file, new_train_file)

