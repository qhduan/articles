# 用TensorFlow构建一个中文分词模型需要几个步骤

## 分析

中文分词方法现在主要有两种类型，一种是jieba这样软件包中用隐马尔科夫（HMM）模型构建的。

另一种就是使用如TensorFlow、PyTorch这类训练深度学习序列标注（Tagging）模型。

这里面我们主要以后者为参考。

中文分词往往有两个直接目标，一个是把词汇分开，另一个给每个词以词性，例如动词、名词，或者更细分的地点名词、机构名词等等。

如果只是分词，是中文分词任务（SEG，Chinese Segmentation），如果需要词性，也可以被称为词性标注任务（POS，Part os Speech）任务。

这两个任务都可以通过深度学习的序列标注（Tagging）模型实现，当然也可以通过其他模型实现，在文章末尾会提及其他方法。

序列标注是一个将输入序列，标注为另一个符号序列的任务，例如我们定义每个词的开头符号是B，非开头符号是I。

那么分词一句话如：“中文的分词任务”，就可以被标注为“B I B B I B I”，不过具体的颗粒度往往由训练语料决定，例如上一句中，“分词任务”到底是一个词，还是两个词组成的词组，这是由标注决定的。

如果是词性标注（POS），那么上面的序列就需要增加更多符号，例如“你开心吗”，可以被标注为：“Br Ba Ia Bu”

其中“Br”可以认为是一个代词的开头。

“Ba Ia”可以认为是一个形容词的开头和中间部分。

“Bu”是助词的开头。

以此讲一个句子的每个字符（字）都标注为一个新的符号序列，我们就可以得到句子的分词或词性标注了。

## 数据

从理想上来说，我们当然希望能有一个模型直接同时完成分词和词性标注两个任务。

但是现实中可能有一定困难，因为并不是我们能找到的所有数据集都包括了这两者的标注的，也就是有一些数据集可能只标注了分词，有些数据集标注了分词和词性。

这里我们从hankcs整理的一些历史上的分词和词性标注数据集为开始，地址为
[https://github.com/hankcs/multi-criteria-cws](https://github.com/hankcs/multi-criteria-cws)

在这里REPO里，有些数据集是标注了词性的，例如CNC（data/other/cnc/test.txt），但是其他很多数据集是没标注词性的。

这里我们使用一个半监督学习方法（Semi-supervised learning），先用有POS数据的数据集，训练一个模型。

然后用这个模型标注其他只有分词的数据集，扩充这些数据集，最终得到一个融合多个数据集的词性标注模型。

在标注其他数据集的时候，应该注意不影响其他数据集的分词结果，只是给这些分词结果一个词性而已。

注意，所谓半监督学习（Semi-supervised learning），其实是一大类算法、方法的统称，这里使用的方法只是某种非常简单的半监督学习方法的应用。

## 模型

在模型上，我们选择使用Albert-small版本的模型，这个版本的模型大小不到30MB，适合比较轻量级的任务，我们可以先尝试实现一个最简单的序列标注模型。

这样的模型最简单是什么样子呢？大概这样：

```python
import tensorflow as tf
import tensorflow_hub as hub

model = tf.keras.Sequential([
    hub.KerasLayer(
        'https://code.aliyun.com/qhduan/bert_v3/raw/98354b8924d9b22fd7b9c4466e36ae9d42cc5a42/albert_base.tar.gz',
        output_key='sequence_output'
    ),
    tf.keras.layers.Dense(2),
    tf.keras.layers.Activation('softmax')
])
```

以上的模型本质上就实现了序列标注，这里使用的TensorFlow Hub来载入一个bert模型，这个模型的介绍可以参考：
[https://github.com/qhduan/bert-model](https://github.com/qhduan/bert-model)

Dense层提供到符号的转换，例如这里我们只考虑B和I两种符号，这里就可以是2。

我们可以简单测试一下这个模型：

```python
x = tf.constant([
    ['你', '好', '吗'],
    ['你', '好', '']
])
y = tf.constant([
    [0, 1, 0],
    [0, 1, 0]
])

model(x)
```

最后的`model(x)`会输出：

```python
<tf.Tensor: shape=(2, 3, 2), dtype=float32, numpy=
array([[[0.7527009 , 0.24729903],
        [0.7459688 , 0.2540312 ],
        [0.73734486, 0.26265517]],

       [[0.6320372 , 0.36796272],
        [0.71670467, 0.28329533],
        [0.5       , 0.5       ]]], dtype=float32)>
```

这代表了模型的结果，已经输出了softmax后的两个符号的概率值。

我们还可以进一步测试模型训练，例如：

```python
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam())

model.train_on_batch(x, y)
```

最后这一句就可以输出loss。

以上只要我们喂给模型类似上面x、y这样定义的数据，就可以训练相应的分词模型了。

## 技巧

以上我们实现了一个非常简单的序列标注模型。

这里我们使用非常简单的线性层作为输出，在现在的序列标注模型，输出层可能有以下几种：

- 线性层
- RNN + 线性层
- CRF
- RNN + CRF
- MRC

线性层就比较简单，如我们上面所写。

CRF层的话可以参考TensorFlow Addons的CRF实现：
[https://www.tensorflow.org/addons/api_docs/python/tfa/text/crf](https://www.tensorflow.org/addons/api_docs/python/tfa/text/crf)

MRC是指机器学习理解，这个方法也是可以用来进行分词、命名实体识别（NER）等工作的，不过在分词上不常用。

## 打包

TensorFlow 2.x的一个优势是，在比较多平台比较方便扩展，至少在Python、C++、NodeJS这个角度。

所以就可以很方便的把当前的TensorFlow模型打包为应用，具体例子可以参考本文这个分词的例子：

[https://github.com/deepdialog/tfseg](https://github.com/deepdialog/tfseg)

安装：`pip install tfseg`

分词：

```python
>>> import tfseg
>>> tfseg.lcut('我爱北京天安门')
['我', '爱', '北京', '天安门']
```

词性：

```python
>>> from tfseg import posseg
>>> posseg.lcut('我爱北京天安门')
[pair('我', 'r'), pair('爱', 'v'), pair('北京', 'ns'), pair('天安门', 'ns')]
>>> posseg.lcut('我爱北京天安门')[0].word
'我'
>>> posseg.lcut('我爱北京天安门')[0].flag
'r'
```
