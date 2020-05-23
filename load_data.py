#!/usr/bin/env python 
# -*- coding:utf-8 -*-
'''
根据xml生成训练样本
'''
import xml.etree.ElementTree as ET

import pandas as pd
import os
import sys


#遍历xml文件
def traverseXml(element):
    # print (len(element))
    if len(element)>0:
        for child in element:
            print (child.tag, "----", child.attrib)
            traverseXml(child)


def convert_xml_to_csv(xml_file, train_data_file, is_test = False):
    '''
    解析xml文件，存入df，并保存为csv文件
    保留尽可能多的原数据信息，保存成dataframe的格式，然后保存到csv文件里面
    dataframe格式的表头为number,qid,label,question1,question2

    文件格式：
    <?xml version="1.0" encoding="utf-8"?>
	<TrainCorpus>
		<Questions number="0">
			<EquivalenceQuestions>
				<question>哪些情形下，不予受理民事诉讼申请？</question>
				<question>民事诉讼中对哪些情形的起诉法院不予受理</question>
				<question>人民法院不予受理的民事案件有哪些情形？</question>
			</EquivalenceQuestions>
			<NotEquivalenceQuestions>
				<question>民事诉讼什么情况下不能立案</question>
				<question>哪些案件会给开具民事诉讼不予立案通知书</question>
				<question>法院对于哪些案件再审申请不予受理</question>
			</NotEquivalenceQuestions>
			...

    :param xml_file:
    :param train_data_file:dataframe的保存位置
    :param is_test:是否是测试集，默认为不是测试集
    :return:
    '''
    # print(os.path.isfile(xml_file))
    '''
    需要先把train_set.xml文件打开，把首行的utf8改成utf-8，不然python没法解析
    否则报错xml.etree.ElementTree.ParseError: not well-formed (invalid token): line 5, column 14
    '''
    tree = ET.parse(xml_file)
    print("tree type:", type(tree))

    # 用list保存dataframe中每一列的内容
    number_list = []
    qid_list = []
    label_list = []
    question1_list = []
    question2_list = []

    # 获得根节点
    root = tree.getroot()
    # print("root type:", type(root))
    # print(root.tag, "----", root.attrib)

    qid = 0
    for Questions in root:  # <Element 'Questions' at 0x0000024FDEE32E58>
        # print("遍历Questions", Questions.tag, "----", Questions.attrib)
        number = Questions.attrib
        for category in Questions:  # <Element 'EquivalenceQuestions' at 0x0000024FDEEBC598>
            '''保存EquivalenceQuestions或NotEquivalenceQuestions的问题，然后排列组合构成数据集'''
            question_list = []
            for question in category:
                question_str = question.text
                question_list.append(question_str)

            for i in range(len(question_list)):
                for j in range(i+1, len(question_list)):
                    # 对于一半的数据：调换一下question1和question2的顺序，防止模型学到无用的信息
                    if qid % 2 == 0:
                        question1_list.append(question_list[i])
                        question2_list.append(question_list[j])
                    else:
                        question1_list.append(question_list[j])
                        question2_list.append(question_list[i])
                    qid_list.append(qid)
                    qid += 1
                    number_list.append(number['number'])
                    if category.tag == "EquivalenceQuestions":
                        label_list.append(1)
                    if category.tag == "NotEquivalenceQuestions":
                        label_list.append(0)

    train_dict = {"number":number_list, "qid":qid_list, "label":label_list , "question1":question1_list , "question2":question2_list}  # 将列表a，b转换成字典
    train_df = pd.DataFrame(train_dict)  # 将字典转换成为数据框
    from sklearn.utils import shuffle
    train_df = shuffle(train_df)
    train_df[['qid', 'label', 'question1', 'question2']].to_csv(train_data_file, index=None, encoding='utf-8', sep='\t')


# 比赛讨论区生成数据方式
#coding=utf-8
from xml.dom.minidom import parse

def generate_train_data_pair(equ_questions, not_equ_questions):
    # qid	label	question1	question2  qid实际上没有必要，但是不想改其他程序了，就保留下来了，统一赋值为2
    a = ["2"+"\t" +"0"+'\t' +x+"\t" +y for x in equ_questions for y in not_equ_questions]
    b = ["2"+"\t" +"1"+'\t' +x+"\t" +y for x in equ_questions for y in equ_questions if x!=y]
    return a+b

def parse_train_data(xml_data):
    pair_list = []
    doc = parse(xml_data)
    collection = doc.documentElement
    for i in collection.getElementsByTagName("Questions"):
        # if i.hasAttribute("number"):
        #     print ("Questions number=", i.getAttribute("number"))
        EquivalenceQuestions = i.getElementsByTagName("EquivalenceQuestions")
        NotEquivalenceQuestions = i.getElementsByTagName("NotEquivalenceQuestions")
        equ_questions = EquivalenceQuestions[0].getElementsByTagName("question")
        not_equ_questions = NotEquivalenceQuestions[0].getElementsByTagName("question")
        equ_questions_list, not_equ_questions_list = [], []
        for q in equ_questions:
            try:
                equ_questions_list.append(q.childNodes[0].data.strip())
            except:
                continue
        for q in not_equ_questions:
            try:
                not_equ_questions_list.append(q.childNodes[0].data.strip())
            except:
                continue
        pair = generate_train_data_pair(equ_questions_list, not_equ_questions_list)
        pair_list.extend(pair)
    print("All pair count=", len(pair_list))
    return pair_list
def write_train_data(file, pairs):
    with open(file, "w") as f:
        f.write('qid'+'\t'+'label'+'\t'+'question1'+'\t'+'question2'+'\n')
        for pair in pairs:
            f.write(pair+"\n")


if __name__ == '__main__':
    # 自己写的
    # convert_xml_to_csv('data/train_set.xml','data/train.csv')

    # # 比赛讨论区生成数据方式，这个效果好像好一些
    pair_list = parse_train_data("data/train_set.xml")
    write_train_data("data/new_train_data.txt", pair_list)
