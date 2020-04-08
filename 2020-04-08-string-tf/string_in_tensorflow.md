# 让Tensorflow直接输入字符串，无需额外词表的3种方法

tensorflow.strings是很早就加入的内容，不过直到2.1/2.2版本才开始有越来越好的支持，而它诞生的目的，当然官方应该没提，我觉得就是为了让模型真正的实现End-to-End，至少在运行时无需额外的词表，那么是不是可以不用词表呢，答案当然是Yes，但是也有一定的代价。

这样做的好处就是，模型迁移、打包、发布的时候，不需要额外的词表处理的程序，或者直接可以用类似tensorflow-hub的方式发布，而避免了自定义的词表文件等等

## 第一种方法，把hash当作词表

第一种方法，然后把每个字（词）利用tf.strings.to_hash_bucket_fast进行hasing，编码到一个具体的索引（int）上

这种方法的主要问题是，这个hashing方法还是很容易冲突的，而为了避免冲突就要用很大的词表，所以这种方法并不是很推荐


```python
import tensorflow as tf
import tensorflow_datasets as tfds
print(tf.__version__)
```

    2.2.0-rc2


这里安装的是笔者很久前封装的一个情感数据集


```python
!pip install zh_dataset_inews > /dev/null
```


```python
from zh_dataset_inews import title_train, label_train
from zh_dataset_inews import title_dev, label_dev
```

这里看到我们就把标题字符串当作x输入，标签三种情感作为输出


```python
print(list(zip(
    title_train[:5], label_train[:5])))
```

    [('周六晚到卖场听夜场摇滚', 1), ('北京老教授泄露，持有山河药辅节后下跌公告，速速看看！！！', 1), ('张滩镇积极开展基干民兵训练活动', 0), ('俩小伙无证骑摩托，未成年还试图闯卡！', 2), ('不好意思，你不配做深圳人!_搜狐汽车_搜狐网', 2)]


这里可以看到，string是可以直接被转换为tensor的


```python
x = tf.constant(title_train[:2])
print(x)
```

    tf.Tensor(
    [b'\xe5\x91\xa8\xe5\x85\xad\xe6\x99\x9a\xe5\x88\xb0\xe5\x8d\x96\xe5\x9c\xba\xe5\x90\xac\xe5\xa4\x9c\xe5\x9c\xba\xe6\x91\x87\xe6\xbb\x9a'
     b'\xe5\x8c\x97\xe4\xba\xac\xe8\x80\x81\xe6\x95\x99\xe6\x8e\x88\xe6\xb3\x84\xe9\x9c\xb2\xef\xbc\x8c\xe6\x8c\x81\xe6\x9c\x89\xe5\xb1\xb1\xe6\xb2\xb3\xe8\x8d\xaf\xe8\xbe\x85\xe8\x8a\x82\xe5\x90\x8e\xe4\xb8\x8b\xe8\xb7\x8c\xe5\x85\xac\xe5\x91\x8a\xef\xbc\x8c\xe9\x80\x9f\xe9\x80\x9f\xe7\x9c\x8b\xe7\x9c\x8b\xef\xbc\x81\xef\xbc\x81\xef\xbc\x81'], shape=(2,), dtype=string)


这里用tf.strings.unicode_split进行按字分开，如果是英文也可以用tf.strings.split按其他字符分开，特殊情况也可以用tf.strings.regex_replace这样的正则替换处理

注意下面的类型是tf.RaggedTensor，而不是普通的Tensor，Ragged是不定长的，这个是有问题的，后续我们会通过.to_tensor转换回来


```python
x = tf.keras.layers.Lambda(lambda x: tf.strings.unicode_split(x, 'UTF-8'))(x)
print(x)
```

    <tf.RaggedTensor [[b'\xe5\x91\xa8', b'\xe5\x85\xad', b'\xe6\x99\x9a', b'\xe5\x88\xb0', b'\xe5\x8d\x96', b'\xe5\x9c\xba', b'\xe5\x90\xac', b'\xe5\xa4\x9c', b'\xe5\x9c\xba', b'\xe6\x91\x87', b'\xe6\xbb\x9a'], [b'\xe5\x8c\x97', b'\xe4\xba\xac', b'\xe8\x80\x81', b'\xe6\x95\x99', b'\xe6\x8e\x88', b'\xe6\xb3\x84', b'\xe9\x9c\xb2', b'\xef\xbc\x8c', b'\xe6\x8c\x81', b'\xe6\x9c\x89', b'\xe5\xb1\xb1', b'\xe6\xb2\xb3', b'\xe8\x8d\xaf', b'\xe8\xbe\x85', b'\xe8\x8a\x82', b'\xe5\x90\x8e', b'\xe4\xb8\x8b', b'\xe8\xb7\x8c', b'\xe5\x85\xac', b'\xe5\x91\x8a', b'\xef\xbc\x8c', b'\xe9\x80\x9f', b'\xe9\x80\x9f', b'\xe7\x9c\x8b', b'\xe7\x9c\x8b', b'\xef\xbc\x81', b'\xef\xbc\x81', b'\xef\xbc\x81']]>


这里是重点了，将上面的字符串hashing到integer


```python
vocab_size_max = 100000
x = tf.keras.layers.Lambda(
    lambda x: tf.strings.to_hash_bucket_fast(x, vocab_size_max - 1) + 1
)(x)
print(x)
```

    <tf.RaggedTensor [[3737, 81563, 36492, 8802, 6236, 73441, 33783, 21987, 73441, 50103, 33946], [30567, 29590, 53848, 95638, 50899, 61624, 43888, 51168, 29600, 80263, 21838, 67875, 14526, 30253, 41754, 37904, 5095, 86897, 22536, 45360, 51168, 88374, 88374, 95206, 95206, 21442, 21442, 21442]]>


转换回tensor，默认补齐0


```python
x = tf.keras.layers.Lambda(lambda x: x.to_tensor())(x)
print(x)
```

    tf.Tensor(
    [[ 3737 81563 36492  8802  6236 73441 33783 21987 73441 50103 33946     0
          0     0     0     0     0     0     0     0     0     0     0     0
          0     0     0     0]
     [30567 29590 53848 95638 50899 61624 43888 51168 29600 80263 21838 67875
      14526 30253 41754 37904  5095 86897 22536 45360 51168 88374 88374 95206
      95206 21442 21442 21442]], shape=(2, 28), dtype=int64)


下面就是普通流程了，走embedding和lstm或其他算法


```python
x = tf.keras.layers.Embedding(vocab_size_max, 32, mask_zero=True)(x)
```


```python
x = tf.keras.layers.LSTM(32)(x)
print(x)
```

    tf.Tensor(
    [[-0.00434301  0.00475923 -0.0013128  -0.00217107 -0.01150594 -0.00142152
       0.01113657  0.00575013 -0.01967175 -0.00572485  0.0034816   0.00513697
       0.00402082  0.00214773 -0.00099676 -0.00361315  0.00678926 -0.00452719
       0.00601438 -0.00380107 -0.00611966  0.00461597  0.00622617 -0.00226157
      -0.00505114  0.00228177 -0.00762874  0.01238765  0.00567317 -0.00654116
      -0.01136432 -0.00287258]
     [-0.00782042  0.00517442 -0.01557797 -0.0235881  -0.00743411  0.00572338
      -0.00213397 -0.00319536 -0.00209199 -0.00467343 -0.00185322  0.00279626
       0.02131106 -0.01428989 -0.0063082   0.01414087  0.00075325  0.00069798
       0.02238634  0.01145286 -0.01497656 -0.00505382 -0.01168405 -0.00315634
      -0.00020887  0.01178367 -0.00746859  0.01161212  0.01726119 -0.00421635
      -0.03153729 -0.01538917]], shape=(2, 32), dtype=float32)



```python
x = tf.keras.layers.Dense(3, activation='softmax')(x)
print(x)
```

    tf.Tensor(
    [[0.33561605 0.3327882  0.33159578]
     [0.32923535 0.33757624 0.3331885 ]], shape=(2, 3), dtype=float32)


完整的一个model构建是这样的

注意多了一行tf.squeeze，因为之后我们会把输入从`[str1, str2]`转换为`[[str1], [str2]]`，因为前者相当于一个`[None,]`的shape，后者相当于`[1, None]`的shape，不这样做会让tensorflow无法对齐输入


```python
vocab_size_max = 1000000
input_layer = tf.keras.layers.Input(shape=(1,), dtype='string')
x = input_layer
x = tf.keras.layers.Lambda(lambda x: tf.strings.unicode_split(x, 'UTF-8'))(x)
x = tf.keras.layers.Lambda(
    lambda x: tf.strings.to_hash_bucket_fast(x, vocab_size_max - 1) + 1
)(x)
x = tf.keras.layers.Lambda(lambda x: x.to_tensor())(x)
x = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, 1))(x)  # 多了一个这个
x = tf.keras.layers.Embedding(vocab_size_max, 32, mask_zero=True)(x)
x = tf.keras.layers.LSTM(32)(x)
x = tf.keras.layers.Dense(3, activation='softmax')(x)
model = tf.keras.Model(inputs=input_layer, outputs=x)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)
```


```python
model.fit(
    tf.constant([[x] for x in title_train]),
    tf.constant(label_train),
    epochs=10,
    validation_data=(
        tf.constant([[x] for x in title_dev]),
        tf.constant(label_dev)
    )
)
```

    Epoch 1/10
    168/168 [==============================] - 89s 532ms/step - loss: 0.8764 - accuracy: 0.5974 - val_loss: 0.6422 - val_accuracy: 0.7477
    Epoch 2/10
    168/168 [==============================] - 88s 522ms/step - loss: 0.5481 - accuracy: 0.7899 - val_loss: 0.5881 - val_accuracy: 0.7598
    Epoch 3/10
    168/168 [==============================] - 88s 523ms/step - loss: 0.4215 - accuracy: 0.8476 - val_loss: 0.6099 - val_accuracy: 0.7528
    Epoch 4/10
    168/168 [==============================] - 87s 520ms/step - loss: 0.3354 - accuracy: 0.8820 - val_loss: 0.6381 - val_accuracy: 0.7447
    Epoch 5/10
    168/168 [==============================] - 87s 521ms/step - loss: 0.2772 - accuracy: 0.9107 - val_loss: 0.7183 - val_accuracy: 0.7457
    Epoch 6/10
    168/168 [==============================] - 88s 523ms/step - loss: 0.2420 - accuracy: 0.9193 - val_loss: 0.7684 - val_accuracy: 0.7447
    Epoch 7/10
    168/168 [==============================] - 88s 524ms/step - loss: 0.2105 - accuracy: 0.9287 - val_loss: 0.8857 - val_accuracy: 0.7137
    Epoch 8/10
    168/168 [==============================] - 88s 521ms/step - loss: 0.1802 - accuracy: 0.9417 - val_loss: 0.9204 - val_accuracy: 0.7357
    Epoch 9/10
    168/168 [==============================] - 88s 523ms/step - loss: 0.1602 - accuracy: 0.9505 - val_loss: 1.0239 - val_accuracy: 0.7217
    Epoch 10/10
    168/168 [==============================] - 88s 522ms/step - loss: 0.1535 - accuracy: 0.9529 - val_loss: 0.9698 - val_accuracy: 0.7167





    <tensorflow.python.keras.callbacks.History at 0x7f61b49d24e0>



注意这里是在测试模型保存和读取，Tensorflow现在的问题还是很多，经常会出现一个模型能训练，但是不能保存，或者能保存但是不能读取的情况，所以这些都是必要的测试手段


```python
model.save('/tmp/str0')
```

    INFO:tensorflow:Assets written to: /tmp/str0/assets


    INFO:tensorflow:Assets written to: /tmp/str0/assets



```python
tf.keras.models.load_model('/tmp/str0')
```




    <tensorflow.python.keras.engine.training.Model at 0x7f61b0d112e8>



## 第二种方法，使用tf.lookup

tf.lookup就类似常规的词表构建方法，它需要我们自定义一个词表层，将之加入模型的某一层


```python
from collections import Counter
```

快速的构建一下词表文件


```python
c = Counter()
for t in title_train:
    c.update(list(t))
keys = list(c.keys())
```

这里用tf.lookup.StaticHashTable函数构建一个映射表


```python
class VocabLayer(tf.keras.layers.Layer):
    def __init__(self, keys, **kwargs):
        super(VocabLayer, self).__init__(**kwargs)
        vals = list(range(1, len(keys) + 1))
        keys = tf.constant(keys)
        vals = tf.constant(vals)
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, vals), 0)
    
    def call(self, inputs):
        return self.table.lookup(inputs)
```

完整模型就这样：


```python
input_layer = tf.keras.layers.Input(shape=(1,), dtype='string')
x = input_layer
x = tf.keras.layers.Lambda(lambda x: tf.strings.unicode_split(x, 'UTF-8'))(x)
x = tf.keras.layers.Lambda(lambda x: x.to_tensor())(x)
x = VocabLayer(keys)(x)
x = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, 1))(x)  # 多了一个这个
x = tf.keras.layers.Embedding(len(keys) + 1, 32, mask_zero=True)(x)
x = tf.keras.layers.LSTM(32)(x)
x = tf.keras.layers.Dense(3, activation='softmax')(x)
model = tf.keras.Model(inputs=input_layer, outputs=x)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)
```


```python
model.fit(
    tf.constant([[x] for x in title_train]),
    tf.constant(label_train),
    epochs=10,
    validation_data=(
        tf.constant([[x] for x in title_dev]),
        tf.constant(label_dev)
    )
)
```

    Epoch 1/10
    168/168 [==============================] - 5s 29ms/step - loss: 0.8727 - accuracy: 0.5903 - val_loss: 0.6735 - val_accuracy: 0.7307
    Epoch 2/10
    168/168 [==============================] - 4s 24ms/step - loss: 0.5573 - accuracy: 0.7806 - val_loss: 0.6018 - val_accuracy: 0.7598
    Epoch 3/10
    168/168 [==============================] - 4s 24ms/step - loss: 0.4275 - accuracy: 0.8476 - val_loss: 0.6293 - val_accuracy: 0.7457
    Epoch 4/10
    168/168 [==============================] - 4s 25ms/step - loss: 0.3639 - accuracy: 0.8639 - val_loss: 0.6166 - val_accuracy: 0.7538
    Epoch 5/10
    168/168 [==============================] - 4s 24ms/step - loss: 0.2906 - accuracy: 0.9055 - val_loss: 0.7207 - val_accuracy: 0.7558
    Epoch 6/10
    168/168 [==============================] - 4s 24ms/step - loss: 0.2477 - accuracy: 0.9212 - val_loss: 0.8145 - val_accuracy: 0.7307
    Epoch 7/10
    168/168 [==============================] - 4s 24ms/step - loss: 0.2102 - accuracy: 0.9339 - val_loss: 0.8339 - val_accuracy: 0.7437
    Epoch 8/10
    168/168 [==============================] - 4s 24ms/step - loss: 0.1890 - accuracy: 0.9423 - val_loss: 0.8675 - val_accuracy: 0.7327
    Epoch 9/10
    168/168 [==============================] - 4s 24ms/step - loss: 0.1663 - accuracy: 0.9473 - val_loss: 0.9264 - val_accuracy: 0.7297
    Epoch 10/10
    168/168 [==============================] - 4s 24ms/step - loss: 0.1412 - accuracy: 0.9572 - val_loss: 0.9959 - val_accuracy: 0.7447





    <tensorflow.python.keras.callbacks.History at 0x7f61abfaf198>




```python
model.save('/tmp/str1')
```

    INFO:tensorflow:Assets written to: /tmp/str1/assets


    INFO:tensorflow:Assets written to: /tmp/str1/assets



```python
tf.keras.models.load_model('/tmp/str1')
```




    <tensorflow.python.keras.engine.training.Model at 0x7f61a8917da0>



## 第三种方法，直接使用utf-8的编码

在用正则表达式判断字符串是否为中文的时候经常用表达式`[\u4e00-\u9fa5]`，这代表在utf-8编码下，主要中文都是在19968~40869这个范围的。所以我们简单点，把所有50000以下的字符都编码，超过的字符按照50000算，这样最多就50000个词表大小，并不算很大，很多时候中文按字分词就够了

下面最主要的函数是tf.strings.unicode_decode，其他大部分程序与上面都无异


```python
input_layer = tf.keras.layers.Input(shape=(1,), dtype='string')
x = input_layer
x = tf.keras.layers.Lambda(lambda x: tf.strings.unicode_decode(x, 'UTF-8'))(x)
x = tf.keras.layers.Lambda(lambda x: x.to_tensor())(x)
x = tf.keras.layers.Lambda(lambda x: tf.math.minimum(x, 50000))(x)
x = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, 1))(x)  # 多了一个这个
x = tf.keras.layers.Embedding(50000 + 1, 32, mask_zero=True)(x)
x = tf.keras.layers.LSTM(32)(x)
x = tf.keras.layers.Dense(3, activation='softmax')(x)
model = tf.keras.Model(inputs=input_layer, outputs=x)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)
```


```python
model.fit(
    tf.constant([[x] for x in title_train]),
    tf.constant(label_train),
    epochs=10,
    validation_data=(
        tf.constant([[x] for x in title_dev]),
        tf.constant(label_dev)
    )
)
```

    Epoch 1/10
    168/168 [==============================] - 7s 40ms/step - loss: 0.8466 - accuracy: 0.6332 - val_loss: 0.6403 - val_accuracy: 0.7528
    Epoch 2/10
    168/168 [==============================] - 6s 35ms/step - loss: 0.5439 - accuracy: 0.7869 - val_loss: 0.5941 - val_accuracy: 0.7578
    Epoch 3/10
    168/168 [==============================] - 6s 36ms/step - loss: 0.4154 - accuracy: 0.8443 - val_loss: 0.6274 - val_accuracy: 0.7518
    Epoch 4/10
    168/168 [==============================] - 6s 36ms/step - loss: 0.3359 - accuracy: 0.8818 - val_loss: 0.6531 - val_accuracy: 0.7367
    Epoch 5/10
    168/168 [==============================] - 6s 35ms/step - loss: 0.2847 - accuracy: 0.9020 - val_loss: 0.7209 - val_accuracy: 0.7367
    Epoch 6/10
    168/168 [==============================] - 6s 36ms/step - loss: 0.2387 - accuracy: 0.9232 - val_loss: 0.7926 - val_accuracy: 0.7387
    Epoch 7/10
    168/168 [==============================] - 6s 35ms/step - loss: 0.2017 - accuracy: 0.9335 - val_loss: 0.8691 - val_accuracy: 0.7327
    Epoch 8/10
    168/168 [==============================] - 6s 36ms/step - loss: 0.1845 - accuracy: 0.9415 - val_loss: 0.9354 - val_accuracy: 0.7277
    Epoch 9/10
    168/168 [==============================] - 6s 36ms/step - loss: 0.1660 - accuracy: 0.9496 - val_loss: 0.9542 - val_accuracy: 0.7417
    Epoch 10/10
    168/168 [==============================] - 6s 35ms/step - loss: 0.1421 - accuracy: 0.9587 - val_loss: 0.9973 - val_accuracy: 0.7317





    <tensorflow.python.keras.callbacks.History at 0x7f61a8369588>




```python
model.save('/tmp/str2')
```

    INFO:tensorflow:Assets written to: /tmp/str2/assets


    INFO:tensorflow:Assets written to: /tmp/str2/assets



```python
tf.keras.models.load_model('/tmp/str2')
```




    <tensorflow.python.keras.engine.training.Model at 0x7f61b591a630>



以上的模型并不是最好，主要是证明能做到，并且做得好了其实是会方便模型的使用者，另一方面是很多项目可以作为快速的baseline，而避免额外的词表之类的程序，给人一个直接可以tf.keras.models.load_model的模型，总比丢一个程序给别人更好，因为它的接口相当于已经确定来，可以降低沟通成本


```python

```
