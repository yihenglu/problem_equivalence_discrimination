import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForSequenceClassification, AdamW, WarmupLinearSchedule
import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'

import utils
import layers

EPOCH = 3
LR = 1e-5
BATCH_SIZE = 12
print('***************v1:3 robert_wwm_large_ext_f5k_epoch3_lr1e5_ml64_bs12***************')

WARMUP_STEPS = 100
T_TOTAL = 1000
FOLD = 5
USE_GPU = True

# 添加变量，控制程序流程
do_trian = True
do_test = True

def train(fold_all):
    config = BertConfig.from_pretrained('../../model_lib/robert/pytorch/chinese_roberta_wwm_large_ext_pytorch/bert_config.json')
    # config = BertConfig.from_pretrained('../../model_lib/bert/pytorch/xs/bert_config.json')
    # bert-base-chinese
    # config = BertConfig.from_pretrained('../../model_lib/bert/pytorch/xs/bert_config.json')

    print('开始训练...')
    for fold_index in range(FOLD):
        # set fold parameter
        BEST_F1 = 0
        BEST_EPOCH = 0
        loss_list = []
        f1_list = []
        flag = 0

        print('正在加载模型...')
        if USE_GPU:
            model = BertForSequenceClassification.from_pretrained('../../model_lib/robert/pytorch/chinese_roberta_wwm_large_ext_pytorch/', config=config).cuda()
            # model = BertForSequenceClassification.from_pretrained('../../model_lib/bert/pytorch/xs/', config=config).cuda()
        else:
            model = BertForSequenceClassification.from_pretrained('../../model_lib/robert/pytorch/chinese_roberta_wwm_large_ext_pytorch/', config=config)
            # model = BertForSequenceClassification.from_pretrained('../../model_lib/bert/pytorch/xs/', config=config)
        optimizer = AdamW(model.parameters(), lr=LR, correct_bias=False)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps = WARMUP_STEPS, t_total = T_TOTAL)  # T_TOTAL？

        # 制作交叉验证的数据集
        train_list = []
        for _ in range(FOLD):
            # 真的交叉验证
            # if _ != fold_index:
            #     train_list = train_list + fold_all[_]
            # 用上所有数据进行训练，假的交叉验证
            train_list = train_list + fold_all[_]
        dev_list = fold_all[fold_index]

        train_bert_list = utils.bert_input(train_list)
        dev_bert_list = utils.bert_input(dev_list)
        train_dataset = layers.Train_Dataset(train_bert_list)
        dev_dataset = layers.Train_Dataset(dev_bert_list)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        for epoch in range(EPOCH):
            start_time = time.time()
            model.train()
            for text, label in train_dataloader:
                # 转text label为tensor
                text = [sub_text.tolist() for sub_text in text]
                label = [int(sub_label) for sub_label in label]
                if USE_GPU:
                    text = torch.tensor(text).t().cuda()  # 为什么要转置？
                    label = torch.tensor(label).cuda()
                else:
                    text = torch.tensor(text).t()
                    label = torch.tensor(label)                 
                
                # 输入模型
                outputs = model(text, labels=label)
                loss, logits = outputs[:2]

                # 优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                

                # 存储单批次f1 loss
                f1 = utils.batch_f1(logits, label)
                f1_list.append(f1)
                loss_list.append(loss.item())
                flag += 1

                # 输出f1 loss
                if flag % 200 == 0:
                    f1_mean = np.mean(f1_list)
                    loss_mean = np.mean(loss_list)
                    f1_list = []
                    loss_list = []
                    print('fold: {} | epoch: {} | f1: {} | loss: {}'.format(fold_index, epoch, f1_mean, loss_mean))

            # 验证集，每个epoch验证一次
            f1_val = val(model, dev_dataloader)

            print('***********************************************************************')
            print('fold: {} | epoch: {} | 验证集F1值: {}'.format(fold_index, epoch, f1_val))      
            if f1_val > BEST_F1:
                BEST_F1 = f1_val
                BEST_EPOCH = epoch
                torch.save(model, 'robert_wwm_large_ext_f5k_epoch3_lr1e5_ml64_bs12_' + str(fold_index) + 'k_' + 'best_model.m')
                # torch.cuda.empty_cache()
            print('fold: {} | 验证集最优F1值: {}'.format(fold_index, BEST_F1))
            print('fold: {} | 验证集最优epoch: {}'.format(fold_index, BEST_EPOCH))
            print('***********************************************************************')
            end_time = time.time()
            print('epoch cost time:',end_time-start_time)

def val(model, dev_dataloader):
    print('正在验证...')
    f1_list = []
    model.eval()
    for text, label in dev_dataloader:
        text = [sub_text.tolist() for sub_text in text]
        label = [int(sub_label) for sub_label in label]
        if USE_GPU:
            text = torch.tensor(text).t().cuda()
            label = torch.tensor(label).cuda()
        else:
            text = torch.tensor(text).t()
            label = torch.tensor(label)          
        with torch.no_grad():
            # 输入模型
            outputs = model(text)
            logits = outputs[0]
            f1 = utils.batch_f1(logits, label)
            f1_list.append(f1)

    f1_mean = np.mean(f1_list)
    return f1_mean

# 测试
def test(test_dataloader):
    print('开始测试...')

    fold_result = []
    fold_logits_result = []
    for fold_index in range(FOLD):
        model = torch.load('robert_wwm_large_ext_f5k_epoch3_lr1e5_ml64_bs12_' + str(fold_index) + 'k_' + 'best_model.m')
        result_list = []
        result_logits_list = []
        model.eval()
        for text in test_dataloader:
            text = [sub_text.tolist() for sub_text in text[0]]
            if USE_GPU:
                text = torch.tensor(text).t().cuda()
            else:
                text = torch.tensor(text).t()

            with torch.no_grad():
                outputs = model(text)
                logits = outputs[0]
                # 保存logits
                for item in logits.tolist():
                    result_logits_list.append(item)
                label = logits.argmax(dim=1).tolist()
                for item in label:
                    result_list.append(item)

        fold_result.append(result_list)
        fold_logits_result.append(result_logits_list)
        # torch.cuda.empty_cache()
    
    return fold_result, fold_logits_result

if __name__ == "__main__":
    # print('正在预处理...')
    fold_all, test_bert_list = utils.main()
    if do_trian:
        train(fold_all)

    # 测试
    if do_test:
        test_dataset = layers.Test_Dataset(test_bert_list)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        fold_list,fold_logits_list = test(test_dataloader)
        vote_result = utils.vote(fold_list)
        vote_logits_result = utils.vote_logits(fold_logits_list)
        utils.write_csv(vote_result)
        print('测试结果保存成功，请提交')