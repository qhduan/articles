# 用NodeJS/TensorFlowJS调用BERT实现文本分类

## 几个提前知识

1. TensorFlowJS可以简单认为有Browser和NodeJS两个版本，前者可以运行在浏览器，后者可以运行在NodeJS环境下
1. NodeJS版本的模型推理速度比Python快哦！参考[官方这个博客 https://blog.tensorflow.org/2020/05/how-hugging-face-achieved-2x-performance-boost-question-answering.html](https://blog.tensorflow.org/2020/05/how-hugging-face-achieved-2x-performance-boost-question-answering.html)
1. NodeJS版本理论上也是可以用GPU的
1. 文本以NodeJS为基础，给出一个文本分类例子œ
1. 按照当前的情况，NodeJS版本其实更适合直接调用Python训练好的模型使用，因为加载速度和推理速度都比Python版本快的原因，如果不是必须要用GPU的话对于小模型更是可以适合配合FaaS等工具更好的实现云AI函数

更多内容和代码可以参考[这个REPO https://github.com/qhduan/bert-model/](https://github.com/qhduan/bert-model/)

## TensorFlowJS/NodeJS的限制

1. 一些算子不支持，例如python版本有的tf.strings.*下面的算子
1. 虽然NodeJS版本可以加载TensorFlow 2.x saved model格式，但是不能继续训练（python是可以的）
1. 训练速度还是比python的慢

## 测试环境准备

数据方面这里我们用之前[ChineseGLUE https://github.com/ChineseGLUE/ChineseGLUE](https://github.com/ChineseGLUE/ChineseGLUE)
的测试数据机LCQMC。这是一个判断两个问题是否等价的数据集，例如“喜欢打篮球的男生喜欢什么样的女生”和“爱打篮球的男生喜欢什么样的女生”等价。

注：新版本ChineseGLUE已经变为[CLUEBenchmark https://github.com/CLUEbenchmark/CLUE](https://github.com/CLUEbenchmark/CLUE)
，并没有这个数据集了。


```bash
$ curl --output train.json https://deepdialog.coding.net/p/dataset/d/dataset/git/raw/master/LCQMC/train.json
$ curl --output dev.json https://deepdialog.coding.net/p/dataset/d/dataset/git/raw/master/LCQMC/dev.json
```

下载中文BERT的词表，几乎所有的中文BERT都是基于最开始谷歌发布的词表的，所以没什么区别

```bash
$ curl --output vocab.txt https://deepdialog.coding.net/p/zh-roberta-wwm/d/zh-roberta-wwm/git/raw/master/vocab.txt
```

下载模型，并解压到bert目录

```bash
$ mkdir -p bert
$ cd bert
$ curl --output bert.tar.gz https://deepdialog.coding.net/p/zh-roberta-wwm/d/zh-roberta-wwm/git/raw/master/zh-roberta-wwm-L12.tar.gz
$ tar xvzf bert.tar.gz
$ cd ..
```

安装Node依赖

```
npm i install @tensorflow/tfjs-node tokenizers
```

## 代码

```JavaScript
const fs = require('fs')
const tf = require('@tensorflow/tfjs-node')
// huggingface的bert分词包
const BertWordPieceTokenizer = require('tokenizers').BertWordPieceTokenizer


/**
 * 构建文本分类模型
 * 输入的是BERT输出的sequence_output序列
 * 输出2分类softmax
 */
function buildModel() {
    const input = tf.input({shape: [null, 768], dtype: 'float32'})
    // 这里之所以用rnn对bert输出序列进行训练，而不是直接针对[CLS]输出进行训练
    // 是因为如果不fine-tune bert的参数的话，只用[CLS]效果会差一点
    const rnn = tf.layers.bidirectional({
        layer: tf.layers.lstm({units: 128, returnSequences: false})
    })
    // masking很重要，我封装的模型padding的部分会输出 0.0 （有可能是 -0.0，但是也可以被mask）
    const mask = tf.layers.masking({maskValue: 0.0})
    const dense = tf.layers.dense({units: 2, activation: 'softmax'})
    const output = dense.apply(rnn.apply(mask.apply(input)))
    const model = tf.model({inputs: input, outputs: output})
    model.compile({
        optimizer: 'adam',
        loss: 'sparseCategoricalCrossentropy',
        metrics: ['acc'],
    })
    return model
}


(async () => {

    // 加载词表/分词和BERT
    const wordPieceTokenizer = await BertWordPieceTokenizer.fromOptions({ vocabFile: "./vocab.txt" })
    const bert = await tf.node.loadSavedModel('./bert')

    // 构建数据流
    // 文本输入会经过tokenizers
    // 然后用bert计算出sequence_output
    // 不更新bert的参数是因为nodejs现在还无法训练读取的模型
    function makeGenerator(objs, batchSize) {
        function* dataGenerator() {
            let xs = []
            let ys = []
            for (const obj of objs) {
                xs.push(obj['tokens'])
                ys.push(Number.parseInt(obj['label']))
                if (xs.length == ys.length && xs.length == batchSize) {
                    // 下面几行，是对数据进行padding到一样长度，补足的部分使用空字符串
                    const maxLength = Math.max.apply(
                        Math,
                        xs.map(x => x.length)
                    )
                    xs = xs.map(x => {
                        while(x.length < maxLength) {
                            x = x.concat([''])
                        }
                        return x
                    })
                    xs = tf.tensor(xs)
                    // 这一步是得到bert的输出结果
                    // 如果输入是dict格式，输出也会是dict格式，可以参考tfjs的源代码
                    // 这一步也可以单独用，就类似bert-as-a-service一样
                    xs = bert.predict({
                        input_1: xs
                    })['sequence_output']
                    ys = tf.tensor(ys)
                    // bert的输出作为文本分类模型的输入(xs)
                    // 标签作为文本分类模型的目标(ys)
                    yield {xs, ys}
                    xs = []
                    ys = []
                }
            }
        }
        return dataGenerator
    }

    // 数据集，格式是jsonl，所以用这种方法读取
    console.log('Read dataset')
    const trainObjs = fs.readFileSync(
        'train.json',
        {encoding: 'utf-8'}
    ).split(/\n/).map(JSON.parse)
    const devObjs = fs.readFileSync(
        'dev.json',
        {encoding: 'utf-8'}
    ).split(/\n/).map(JSON.parse)

    // 这里先对分词，是因为分词是async异步函数，而tensorflowjs的generator不支持异步yield
    console.log('Tokenize train dataset')
    for (const obj of trainObjs) {
        obj['tokens'] = (await wordPieceTokenizer.encode(
            obj['sentence1'], obj['sentence2']
        )).tokens
    }
    console.log('Tokenize dev dataset')
    for (const obj of devObjs) {
        obj['tokens'] = (await wordPieceTokenizer.encode(
            obj['sentence1'], obj['sentence2']
        )).tokens
    }
    console.log('Start training')
    
    const batchSize = 32
    const dsTrain = tf.data.generator(makeGenerator(trainObjs, batchSize)).repeat()
    const dsDev = tf.data.generator(makeGenerator(devObjs, batchSize)).repeat()
    const model = buildModel()
    model.fitDataset(dsTrain, {
        batchesPerEpoch: Math.floor(trainObjs.length / batchSize),
        epochs: 1,
        batch_size: batchSize,
        validationData: dsDev,
        validationBatches: Math.floor(devObjs.length / batchSize),
    })

    model.evaluateDataset(dsDev, {
        batches: Math.floor(devObjs.length / batchSize),
    })

})()
```
