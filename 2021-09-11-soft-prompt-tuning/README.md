# 用mT5模型微调中文分类任务示例

mT5模型是T5模型在多语言数据集C4上的继续训练，T5本身是比较早了，是2019年的一个模型，但是后来又有很多次不同的升级。

mT5模型论文发布自2020年10月。

论文：[mT5: A massively multilingual pre-trained text-to-text transformer](https://arxiv.org/pdf/2010.11934.pdf)

代码：[https://github.com/google-research/multilingual-t5](https://github.com/google-research/multilingual-t5)

根据预训练模型进行Prompt-tuning是最近两年的研究热点，从开始的各种搜索词表的hard-prompt-tuning，到只训练一部分的soft-prompt-tuning，各种如何调整prompt的方法，可以参考这篇论文综述：
[Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing](https://arxiv.org/pdf/2107.13586.pdf)

本文主要介绍其中根据叠加prompt embedding进行soft-prompt-tuning如何实现。

具体实现的主要思路参考了这篇2021年4月的论文[The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/pdf/2104.08691.pdf)而代码主要参考了[https://github.com/kipgparker/soft-prompt-tuning](https://github.com/kipgparker/soft-prompt-tuning)

## 什么是soft-prompt-tuning

首先是prompt tuning，就是我们现在已经有了很多类似bert/gpt/t5这样的预训练模型，但是它们都很大，可能有几十亿或者几百亿个参数。

那这些模型进行一般的fine-tune本身就很耗费资源，而一个公司或者项目可能需要很多不同的任务基于这些模型，每个任务又需要单独部署，也很耗费资源。

有没有什么办法能让训练和部署的资源压力更小呢？

我们通过gpt-2/gpt-3的一些zero-shot/few-shot learning为启发点，就是我们给不同任务不同的提示（prompt），就可以很少的训练就可以在同一个模型完成不同的任务。

最开始，这些prompt就是一些特殊的句子，比如说我们给gpt3的提示是：“1+1=2；2+2=4；4+5=9；5+6=”这样的提示，让模型继续生成，希望能输出5+6的正确答案。

这种基于句子或者具体token的prompt，被称为hard-prompt。

那么我们怎么找到每个任务的最好的prompt呢？当然我们可以人工设计去一点一点尝试，或者干脆穷举，当然也有很多基于不同方法的测试，可以参考上面提到的综述论文。

除了hard-prompt以外，假设我们想要更好的结果，是不是我们可以设计一些特殊的类似token的东西，我们称之为prompt embedding，把它和原来模型本身的embedding对齐。
这样利用self-attention的机制，让模型本身可以读取我们加入的embedding。

而训练，就是只更新我们加入的prompt embedding的梯度，也就是只训练模型的非常小的一部分参数，而不去更新整体模型。

这样显然模型训练用的资源就会更好，部署的时候我们也只需要部署一个原版模型，只需要根据不同任务，插入不同的prompt embedding就好了。

所以我们需要：

1. 想办法在原版模型的embedding中，加入我们的prompt embedding
1. 训练模型保证只训练我们加入的这部分embedding，不训练其他的模型参数

## 分类任务的实现

我们说了我们要做中文的分类任务，mT5这样的encoder-decoder结构其实天然的做的是sequence-to-sequence结构，类似机器翻译/对话聊天之类的

那么分类任务怎么设计的

我们先定位任务为，输入一句（段）中文文本，输出一个三分类的标签，0，1，2。

首先输入中文文本，也就是把中文文本作为encoder的输入肯定没问题。

decoder的输入，也没什么好说的，毕竟我们不是seq2seq任务，不需要特殊的输入。

在代码中的encoder和decoder的输入中，加入了prompt embedding的占位符，是任意token id都可以，反正都会被我们的代码替换掉的。

我们要获取的是decoder最后的输出，并把输出中的一些特殊字符位置，当作我们的三分类结果。

decoder默认肯定会输出一个词表长度的向量，我们只拿其中3个使用，实际代码中我是使用3，4，5，三个特殊token id作为判定三分类的结果。

我们具体计算loss也只计算最后decoder输出的这三个token的概率比较，比如3的概率最大，那么就是分类0，4的大就是分类1，5的大就是分类2。

## 具体实现

首先使用tansformers就可以很方便的去下载和调用谷歌的T5/mT5模型

安装pytorch和transformers，以及分词器（tokenizer）所需的sentencepiece

`pip install transformers SentencePiece torch`

然后就可以执行代码自动从网上下载模型：

```python
from transformers import MT5ForConditionalGeneration, T5Tokenizer

model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")
```

然后我们构建一个替换原版模型的输入器，用来把用于训练的prompt embedding加入到模型。

下面代码主要参考[https://github.com/kipgparker/soft-prompt-tuning](https://github.com/kipgparker/soft-prompt-tuning)这个repo进行修改，因为这个repo的训练时基于GPT的，而我们是基于mT5的，所以整体代码上略有区别，而且这个repo的训练代码也不太完整。

```python
import torch
import torch.nn as nn


class SoftEmbedding(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 10, 
                random_range: float = 0.5,
                initialize_from_vocab: bool = True):
        """这个类用来给模型附加一个用于学习的embedding
        Args:
            wte (nn.Embedding): 这个参数，是预训练模型的embedding，载入进来用来提取一些参数。
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                                  n_tokens, 
                                                                                  random_range, 
                                                                                  initialize_from_vocab))

    def initialize_embedding(self, 
                             wte: nn.Embedding,
                             n_tokens: int = 10, 
                             random_range: float = 0.5, 
                             initialize_from_vocab: bool = True):
        """初始化学习向量
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        # 有两种初始化方式，一种是从预训练模型copy一部分token，进行训练
        # 另一种是随机生成一部分训练
        # 结果上来说区别不大
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)
            
    def forward(self, tokens):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
        # 把我们新加入的固定长度的，用于代表任务的prompt embedding，和实际的embedding合并
        return torch.cat([learned_embedding, input_embedding], 1)
```

然后我们把我们设计的这个类，载入到预训练模型里面去：

```python
n_tokens = 100
s_wte = SoftEmbedding(model.get_input_embeddings(), 
                      n_tokens=n_tokens, 
                      initialize_from_vocab=True)
# 用我们设计的类，替换原来的embedding层
model.set_input_embeddings(s_wte)

if torch.cuda.is_available():
    model = model.cuda()

# 把除了第0个，就是我们要训练的prompt embedding以外的参数，都设置为不需要梯度
parameters = list(model.parameters())
for x in parameters[1:]:
    x.requires_grad = False
```

主要的模型部分代码就是上面的部分，训练过程和详细代码参考[Repo https://github.com/qhduan/mt5-soft-prompt-tuning](https://github.com/qhduan/mt5-soft-prompt-tuning)

我们期望的结果是：

1. 模型训练只更新prompt embedding，不更新模型整体参数
1. 模型的结果，和更新整体模型参数的fine-tune尽可能接近
