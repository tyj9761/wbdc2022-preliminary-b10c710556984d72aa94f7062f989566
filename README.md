## [2022中国高校计算机大赛-微信大数据挑战赛](https://algo.weixin.qq.com/)

### 环境依赖<br>

Python 版本：3.7.6 PyTorch 版本：1.11.0 CUDA 版本：11.6<br>
所需环境在requirements.txt中定义。

### 数据<br>
·仅使用大赛提供的有标注数据（10万）。<br>
·未使用任何额外数据。<br>


### 代码结构<br>
(1) 预训练模型 <br> 
预训练模型使用了huggingface 上提供的 [hfl/chinese-macbert-base](https://huggingface.co/hfl/chinese-macbert-base) 模型和 [hfl/chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext) <br> 
 
(2) 代码结构说明<br>
我们使用了部分开源代码，包括[EMA](https://blog.csdn.net/weixin_42677618/article/details/109778055)和[FGM](https://blog.csdn.net/qq_40176087/article/details/121512229)<br>
inference.py是整个模型的融合和推理的脚本<br>
fold.py 对数据集进行了K折划分<br>
train.py 模型训练的脚本<br>
datahelper.py 数据测试集验证集划分以及分词的脚本<br>
util.py 包含各种功能函数<br>
注：model2中的9个全量模型中，第一个模型选取的是第4轮结果，其余均选取的第5轮结果。<br>

### 运行流程说明<br>
(1)model1中需要运行fold.py进行k折划分，然后运行train0.py进行训练。<br>
(2)在model2中直接运行train1.py和train.py进行训练。<br>
训练完成之后直接运行src/inference.py进行融合以及推理。

### 算法模型介绍<br>
model1模型参考了VLbert的结构，采用的分词方式为是对 title、 asr、ocr分别均匀截断，然后拼接，如 [CLS] + title + [SEP] + asr + [SEP] + ocr + [SEP] 这种形式。<br>
model2模型参考了2021年qq浏览器第一名的方案，采用的分词方式为直接拼接，即title + asr + ocr的形式。<br>
未做模型预训练，进行了多模型融合。<br>

### 模型初赛B榜在线结果<br>
model1单折线下在0.67左右，融合线上在0.682左右.<br>
model2单模线下在0.679左右，全量融合线上在0.6882左右.<br>
最终多模型融合结果线上约在0.688903.<br>





