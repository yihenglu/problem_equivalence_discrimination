# [基于Adversarial Attack的问题等价性判别比赛](https://biendata.com/models/category/3768/L_notebook/)

## 项目说明
使用bert作为baseline，说明预训练模型的基本使用流程。

## 文件说明
* [biendata-2019-DIAC](https://github.com/DefuLi/Biendata-2019-DIAC)：可以运行的一个完整目录（项目），为pytorch版。
    * data_processing.py:自定义数据处理部分，如转换为提交所需的指定格式等。
    * layers.py: 自定义训练集、验证集Dataset
    * main_bert_base_chinese.py: 入口文件，调用的第三方库huggingface的transformers库，使用K折CV
    * utils.py: 数据处理
* data:语料
* other:暂且没用的文件
* bert.py:tf版，使用train.sh进行调用。**存在的问题**：不方便添加自定义日志或者交叉验证，因为训练过程被封装了(estimator.train(input_fn=train_input_fn, max_steps=num_train_steps))。
* data_processing.py:自定义数据处理部分，如转换为提交所需的指定格式等。
* DEA.py:数据探索性分析
* load_data.py: 根据原始语料生成训练数据
* rule.py:人工规则
* stopwords.txt:停用词表
* robert_wwm_large_epoch9_lr1_ml128_bs4_all_cols.csv:模型结果
* train.sh:程序执行入口