# Biomedical Domain BERT
This repository provides fine-tuning codes of a language representation model for the biomedical domain, especially designed for biomedical text mining tasks such as biomedical named entity recognition, relation extraction, question answering, etc. Please refer to the cited paper (http://arxiv.org/abs/1901.08746) for more details.


## Installation
Download pre-trained weights from [Naver GitHub repository for BioBERT pre-trained weights](https://github.com/naver/biobert-pretrained). Make sure to specify the versions of pre-trained weights used in your works. Also, note that this repository is based on the [BERT repository](https://github.com/google-research/bert) by Google.

All the fine-tuning experiments were conducted on RP labtop 4 CPU virtual machine which has 10GB of RAM and 90GB of memory storage. The code was tested with Python3 (Ubuntu 18.04). 

## Datasets
Pre-processed version of benchmark datasets for the following task is provided:
*   **[`Named Entity Recognition`](https://drive.google.com/open?id=1OletxmPYNkz2ltOr9pyT0b0iBtUWxslh)**: (17.3 MB), 8 datasets on biomedical named entity recognition

For details on NER datasets, please see **A Neural Network Multi-Task Learning Approach to Biomedical Named Entity Recognition (Crichton et al. 2017)**.
The source of pre-processed datasets are from https://github.com/cambridgeltl/MTL-Bioinformatics-2016 and https://github.com/spyysalo/s800.


Due to the copyright issue of some datasets, we provide links of those datasets instead:
### NER
*   **[`2010 i2b2/VA`](https://www.i2b2.org/NLP/DataSets/Main.php)**

## Fine-tuning BERT
After downloading one of the pre-trained models from [Naver GitHub repository for BioBERT pre-trained weights](https://github.com/naver/biobert-pretrained), unpack it to any directory you want, which is denoted as `$BERT_DIR`. 

### Named Entity Recognition (NER)
Download and unpack the NER datasets provided above (**[`Named Entity Recognition`](https://drive.google.com/open?id=1OletxmPYNkz2ltOr9pyT0b0iBtUWxslh)**). From now on, `$NER_DIR` indicates a folder for a single dataset which should include `train_dev.tsv`, `train.tsv`, `devel.tsv` and `test.tsv`. For example, `export NER_DIR=./NER_Data/NCBI-disease`. Following command runs fine-tuining code on NER with default arguments.
```
mkdir /tmp/ner_outputs/
python run_ner.py \
    --do_train=true \
    --do_eval=true \
    --do_predict=true \
    --vocab_file=$BERT_DIR/vocab.txt \
    --bert_config_file=$BERT_DIR/bert_config.json \
    --init_checkpoint=$BERT_DIR/model.ckpt-1000000 \
    --num_train_epochs=10.0 \
    --data_dir=$NER_DIR/ \
    --output_dir=/tmp/ner_outputs/
```
You can change the arguments as you want. Once you have trained your model, you can use it in inference mode by using `--do_train=false --do_predict=true` for evaluating `test.tsv`.
The token-level evaluation result will be printed as stdout format. For example, the result for NCBI-disease dataset will be like this:
```
INFO:tensorflow:***** token-level evaluation results *****
INFO:tensorflow:  eval_f = 0.9028707
INFO:tensorflow:  eval_precision = 0.8839457
INFO:tensorflow:  eval_recall = 0.92273223
INFO:tensorflow:  global_step = 2571
INFO:tensorflow:  loss = 25.894125
```
(tips : You should go up a few lines to find the result. It comes before `INFO:tensorflow:**** Trainable Variables ****` )

Note that this result is the token-level evaluation measure while the official evaluation should use the entity-level evaluation measure. 
The results of `python run_ner.py` will be recorded as two files: `token_test.txt` and `label_test.txt` in `output_dir`. 
Use `ner_detokenize.py` in `./biocodes/` to obtain word level prediction file.
```
python biocodes/ner_detokenize.py \
--token_test_path=/tmp/ner_outputs/token_test.txt \
--label_test_path=/tmp/ner_outputs/label_test.txt \
--answer_path=$NER_DIR/test.tsv \
--output_dir=/tmp/ner_outputs
```
This will generate `NER_result_conll.txt` in `output_dir`.
Use `conlleval.pl` in `./biocodes/` for entity-level exact match evaluation results.
```
perl biocodes/conlleval.pl < /tmp/ner_outputs/NER_result_conll.txt
```

The entity-level results for NCBI-disease dataset will be like :
```
processed 24497 tokens with 960 phrases; found: 993 phrases; correct: 866.
accuracy:  98.57%; precision:  87.21%; recall:  90.21%; FB1:  88.68
             MISC: precision:  87.21%; recall:  90.21%; FB1:  88.68  993
``` 
Note that this is a sample run of an NER model. Performance of NER models usually converges at more than 50 epochs (learning rate = 1e-5 is recommended).

## License and Disclaimer
Please see LICENSE file for details. Downloading data indicates your acceptance of the Apache disclaimer.

## Reference Paper

Citaton [the Arxiv paper](http://arxiv.org/abs/1901.08746):

```
@article{lee2019biobert,
  title={BioBERT: a pre-trained biomedical language representation model for biomedical text mining},
  author={Lee, Jinhyuk and Yoon, Wonjin and Kim, Sungdong and Kim, Donghyeon and Kim, Sunkyu and So, Chan Ho and Kang, Jaewoo},
  journal={arXiv preprint arXiv:1901.08746},
  year={2019}
}
```
