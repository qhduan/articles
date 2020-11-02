# NLP在TensorFlow 2.x中的最佳实战

NLP Best Practice with TensorFlow

本文会介绍TensorFlow 2.x在处理NLP任务中的一些工具和技巧，包括：

- tf.keras.layers.experimental.preprocessing.TextVectorization
- tf.strings
- tf.data.experimental.bucket_by_sequence_length
- BERT with strings

## TextVectorization

在完成NLP任务的时候，经常需要把文字（一般是字符串），转换为具体的词向量（或字向量）。

或者说把文字转换为对应的词嵌入（Word Embedding/Token Embedding）。

一般来说我们可能会这么做：制作一个词表，然后写程序把对应的词（字）映射到整数序号，然后就可以使用如`tf.keras.layers.Embedding`层，把这个整数映射到词嵌入。

但是这种做法有一个问题，就是你需要一个额外的程序，和一份此表，才能把文字（字符串）转换为具体的整数序号。

因为需要额外的程序，比如需要把一个TensorFlow保存后的模型传给别人，也同时需要传输给别人这个程序和词表，显然麻烦的多。

有没有一种不需要额外程序和采标的方法呢？TensorFlow新加入的特性`TextVectorization`就是这样的功能。

`TextVectorization`默认输入以空格为分割的字符串，同时它和其他TensorFlow/Keras的层不同，它需要先进行学习，具体的代码如下：

```python
x = [
    '你 好 啊',
    'I love you'
]
# 构建层
text_vector = tf.keras.layers.experimental.preprocessing.TextVectorization()
# 学习词表
text_vector.adapt(x)
# 我们可以通过这种方式获取词表（一个list）
print(text_vector.get_vocabulary())

# 输出：
# ['', '[UNK]', '好', '啊', '你', 'you', 'love', 'i']

# 可以看出结果已经
print(text_vector(x))

# 输出：
# tf.Tensor(
# [[4 2 3]
#  [7 6 5]], shape=(2, 3), dtype=int64)
```

然后就可以把`text_vector`加入一个普通的TensorFlow模型

```python
model = tf.keras.Sequential([
    text_vector,
    tf.keras.layers.Embedding(
        len(text_vector.get_vocabulary()),
        32
    ),
    tf.keras.layers.Dense(2)
])

print(model(x))

# 输出：
# <tf.Tensor: shape=(2, 3, 2), dtype=float32, numpy=
# array([[[-0.01258635, -0.01506722],
#         [-0.02729277, -0.04474692],
#         [ 0.02955768,  0.00149873]],
#        [[ 0.01346388,  0.01626211],
#         [-0.03160518,  0.07346839],
#         [-0.01061894, -0.0035725 ]]], dtype=float32)>
```

## tf.strings是什么

那么TextVectorization是怎么实现的呢？其实我们自己也可以实现这个功能，这就要说到TensorFlow的字符串类型和相关的各种算子。

比如我们可以通过`tf.strings.split`来分割字符串

```python
x = [
    '你 好 啊',
    'I love you'
]
print(tf.strings.split(x))

# 输出：
# <tf.RaggedTensor [[b'\xe4\xbd\xa0', b'\xe5\xa5\xbd', b'\xe5\x95\x8a'], [b'I', b'love', b'you']]>
```

词表怎么实现呢，我们就需要使用`tf.lookup.StaticHashTable`

```python
keys_tensor = tf.constant(['你', '好', '啊'])
vals_tensor = tf.constant([1, 2, 3])

table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1)
print(table.lookup(tf.constant(['你', '好'])))

# 输出：
# tf.Tensor([1 2], shape=(2,), dtype=int32)
```

## 数据对齐：bucket_by_sequence_length

处理图片模型的时候，经常需要将图谱缩放到一个固定的大小，不过对于NLP任务来说，句子长度可是不同的，这虽然也可以通过增加padding的方式，即插入空字符的方式对齐。

但是实际上这样的处理是有一定的问题的，就是效率损失。

这种方式虽然能满足算法，但是实际上无论是LSTM/Transform，其实效率都和句子长度有关。

例如有4个句子，两个是长度2，两个是长度100，假设分成两个批次（batch），第一个批次是两个长度2的句子，第二批次是两个长度100的句子，那算法就只需要计算(2 + 100) 的算力。

但是如果四个句子，把一个长度2的句子和一个长度100的句子凑一起，就需要在每个批次的长度2句子后面插入98个空字符，算法需要的算力就是（100 + 100）的算力。

在TensorFlow中，可以使用`tf.data.Dataset.experimental.bucket_by_sequence_length`自动对齐输入数据。

## 基于BERT的举例：更简易的BERT

BERT模型实际上是有3个输入的：token，mask，type

token是经过分词的字符串，转换为的整数序号。

mask是输入长度的遮盖，是在一个batch有中不同长度的句子时的情况。

type是0或1，是bert训练目标中的第二个NSP任务相对的type embedding所需的。

不过对于大多数情况，其实BERT只需要一个输入，就是字符串。

因为对于BERT很多单句/单文档模型的情况，type只需要单一的0就可以了。

而mask也可以通过字符串本身计算出来，例如是否为空字符串。

这个时候我们就可以使用以上提到的字符串方法，让BERT模型直接输入字符串，配合TensorFlow Hub，这就可以方便很多模型的计算。

当然对于BERT的分词器，就很难简单的直接用上面提到的TextVectorization了，这里需要配合TensorFlow Text。

最简单算法可以简化如下：

```python
# $ pip install tensorflow tensorflow-text tensorflow-hub
import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub
tokenizer = hub.load(
    'https://code.aliyun.com/qhduan/bert_v4/raw/500019068f2c715d4b344c3e2216cef280a7f800/bert_tokenizer_chinese.tar.gz'
)
albert = hub.load(
    'https://code.aliyun.com/qhduan/bert_v4/raw/500019068f2c715d4b344c3e2216cef280a7f800/albert_tiny.tar.gz'
)
out = albert(tokenizer(['你好']))

assert out['sequence_output'].shape == (1, 2, 312)
assert out['pooled_output'].shape == (1, 312)
```
