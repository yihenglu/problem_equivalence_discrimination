# robert large
# python bert.py --task_name=sentiment --do_train=false --do_eval=false --do_predict=true \
# --data_dir=./data/ --vocab_file=../model_lib/robert/tensorflow/roeberta_zh_L-24_H-1024_A-16/vocab.txt  \
# --bert_config_file=../model_lib/robert/tensorflow/roeberta_zh_L-24_H-1024_A-16/bert_config_large.json \
# --init_checkpoint=../model_lib/robert/tensorflow/roeberta_zh_L-24_H-1024_A-16/roberta_zh_large_model.ckpt \
# --max_seq_length=64 --train_batch_size=12 --learning_rate=1e-5 \
# --output_dir=./output_robert_large_epoch9_lr1_ml64_bs12/ --num_train_epochs 9


# robert large xp2
# python bert.py --task_name=sentiment --do_train=true --do_eval=true --do_predict=true \
# --data_dir=./data/ --vocab_file=../model_lib/robert/tensorflow/roeberta_zh_L-24_H-1024_A-16/vocab.txt  \
# --bert_config_file=../model_lib/robert/tensorflow/roeberta_zh_L-24_H-1024_A-16/bert_config_large.json \
# --init_checkpoint=../model_lib/robert/tensorflow/roeberta_zh_L-24_H-1024_A-16/roberta_zh_large_model.ckpt \
# --max_seq_length=128 --train_batch_size=6 --learning_rate=1e-5 \
# --output_dir=./output_robert_large_epoch5_lr1_ml128_bs6/ --num_train_epochs 5


# robert wwm large 1080_7
# python bert.py --task_name=sentiment --do_train=true --do_eval=true --do_predict=true \
# --data_dir=./data/ --vocab_file=../model_lib/robert/tensorflow/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/vocab.txt  \
# --bert_config_file=../model_lib/robert/tensorflow/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_config.json \
# --init_checkpoint=../model_lib/robert/tensorflow/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_model.ckpt \
# --max_seq_length=64 --train_batch_size=12 --learning_rate=1e-5 \
# --output_dir=./output_robert_wwm_large_epoch5_lr1_ml64_bs12/ --num_train_epochs 5


# # robert wwm large 1080_7
# python bert.py --task_name=sentiment --do_train=true --do_eval=true --do_predict=true \
# --data_dir=./data/ --vocab_file=../model_lib/robert/tensorflow/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/vocab.txt  \
# --bert_config_file=../model_lib/robert/tensorflow/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_config.json \
# --init_checkpoint=../model_lib/robert/tensorflow/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_model.ckpt \
# --max_seq_length=128 --train_batch_size=4 --learning_rate=1e-5 --num_train_epochs 9 \
# --output_dir=./output_robert_wwm_large_epoch9_lr1_ml128_bs4/


# robert wwm large xp2
# python bert.py --task_name=sentiment --do_train=true --do_eval=true --do_predict=true \
# --data_dir=./data/ --vocab_file=../model_lib/robert/tensorflow/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/vocab.txt  \
# --bert_config_file=../model_lib/robert/tensorflow/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_config.json \
# --init_checkpoint=../model_lib/robert/tensorflow/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_model.ckpt \
# --max_seq_length=128 --train_batch_size=4 --learning_rate=2e-5 --num_train_epochs 2 \
# --output_dir=./output_robert_wwm_large_epoch2_lr2_ml128_bs4/


# robert wwm large xp2
# python bert.py --task_name=sentiment --do_train=true --do_eval=true --do_predict=true \
# --data_dir=./data/ --vocab_file=../model_lib/robert/tensorflow/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/vocab.txt  \
# --bert_config_file=../model_lib/robert/tensorflow/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_config.json \
# --init_checkpoint=../model_lib/robert/tensorflow/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_model.ckpt \
# --max_seq_length=128 --train_batch_size=4 --learning_rate=2e-5 --num_train_epochs 4 \
# --output_dir=./output_robert_wwm_large_epoch4_lr2_ml128_bs4/


# robert wwm large ep=5 1080_7
# python bert.py --task_name=sentiment --do_train=true --do_eval=true --do_predict=true \
# --data_dir=./data/ --vocab_file=../model_lib/robert/tensorflow/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/vocab.txt  \
# --bert_config_file=../model_lib/robert/tensorflow/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_config.json \
# --init_checkpoint=../model_lib/robert/tensorflow/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_model.ckpt \
# --max_seq_length=128 --train_batch_size=4 --learning_rate=1e-5 --num_train_epochs 5 \
# --output_dir=./output_robert_wwm_large_epoch5_lr1_ml128_bs4/