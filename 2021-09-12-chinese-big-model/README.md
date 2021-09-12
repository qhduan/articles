# 盘点一下今年以来的大规模训练模型

## 模型介绍

### 模型1: CPM 1.0

论文：

PM: A Large-scale Generative Chinese Pre-trained Language Model

https://arxiv.org/pdf/2012.00413.pdf

代码：

https://github.com/TsinghuaAI/CPM

文本作者提供的自以为更容易使用的TensorFlow版本：

https://github.com/deepdialog/CPM-LM-TF2

严格来说CPM模型发布在2020年12月左右，其实很多代码和文件都是2021年才放出来的，所以勉强也算“今年”吧。

CPM这个名字应该是Chinese Pretrained language Model的缩写。自从GPT-3在2020年5月发布之后，我其实一直就很期待中文的类似模型的，所以对于CPM也是抱有了很大的期待。

CPM的大部分代码其实是沿用英伟达的Megatron代码，包括盘古alpha也在Megatron的基础上修改并重新实现了一版（盘古标准版本是用华为自研的MindSpore实现），Megatron的链接： https://github.com/NVIDIA/Megatron-LM

Megatron总体实现了一个跟GPT-3基本一致的decoder模型。沿用Megatron代码的CPM 1.0在模型上也跟GPT-3没有什么区别。

### 模型2: 盘古alpha

论文：

PANGU-α: LARGE-SCALE AUTOREGRESSIVE PRETRAINED CHINESE LANGUAGE MODELS WITH AUTO-PARALLEL COMPUTATION

https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha/raw/branch/master/PANGU-%ce%b1.pdf

代码：

https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha#user-content-%E6%A8%A1%E5%9E%8B%E4%B8%8B%E8%BD%BD

https://git.openi.org.cn/PCL-Platform.Intelligence/Chinese-Megatron

在线测试：

https://pangu-alpha.openi.org.cn/

文本作者提供的自以为更容易使用的TensorFlow版本：

https://github.com/deepdialog/PanGu-alpha-tf

盘古alpha在GPT-3的基础之上，做了修改。

盘古alpha的前N-1层的结构和GPT-3是一样的，在最后一层的时候，使用了一个结构和position embedding一致的，新的query embedding，用它来代替最后一层self-attention的query的输入。

也就是说，其他每层的self-attention的qkv，都是还是普通的输入，而最后一层的self-attention的qkv中的q，输入修改为了query embedding的输出。

pangu自己的代码相对有点复杂，而且原谅我不太熟悉MindSpore，可以参考我写的TensorFlow版本代码的这三部分代码，一个文件，分别在127/261/289行：

https://github.com/deepdialog/PanGu-alpha-tf/blob/f64a8985880b7050e804f205f846c5ff9ae8a5be/tf2gpt/model.py#L127

https://github.com/deepdialog/PanGu-alpha-tf/blob/f64a8985880b7050e804f205f846c5ff9ae8a5be/tf2gpt/model.py#L261

https://github.com/deepdialog/PanGu-alpha-tf/blob/f64a8985880b7050e804f205f846c5ff9ae8a5be/tf2gpt/model.py#L289


### 模型3: CPM 2.0

论文：

CPM-2: Large-scale Cost-effective Pre-trained Language Models

https://arxiv.org/pdf/2106.10715.pdf

代码：

https://github.com/TsinghuaAI/CPM

文本作者提供的，基于huggingface的transformers库的自以为可能容易使用的模型调用代码：

https://github.com/deepdialog/CPM-2.0-GEN

CPM 2.0是智源于2021年6月发布的新模型，整体架构不再是GPT-3，而是改为了encoder-decoder结构的T5

而主要的目标也不是类似GPT-3的 `In context zero/few shot learning`，而是最近一年更火的 `prompt-tuning` （p-tuning）方式，这个在我的账号前一篇文章《用mT5模型微调中文分类任务示例》中也更详细的介绍了。

这个模型使用的是最新版本的T5，或者说mT5模型，在论文中主要对比的模型也是mT5。

### 模型4: EVA

论文：

EVA: An Open-Domain Chinese Dialogue System with Large-Scale Generative Pre-Training

https://arxiv.org/pdf/2108.01547.pdf

代码：

https://github.com/BAAI-WuDao/EVA

文本作者提供的，基于huggingface的transformers库的自以为可能容易使用的模型调用代码：

https://github.com/deepdialog/EVA-GEN

### 模型5: 达摩院PLUG

相关新闻：

https://m.thepaper.cn/baijiahao_12274410

演示地址：

https://nlp.aliyun.com/portal#/BigText_chinese

很遗憾没有找到这个模型的论文，也没有代码等等，只有一些发布的新闻的描述以及一个需要登录的在线测试地址。

从描述上来看，这是一个26B参数的中文模型，似乎采用了类似T5的encoder-decoder模型，不过encoder和decoder的大小不对等，分别是24层和6层。

先训练encoder，再微调训练decoder，尽量保证了模型的encoder有足够能力，也保证了decoder模型在生成任务上还不错。

## 模型总结

### 模型大小对比：

一般模型都用xx Billion（B）代表参数大小，因为模型可能是float32，也可能是float16或者其他格式，外加一些其他原因，所以这个参数大小不能简单理解为文件大小。

注意1B是10亿。

CPM 1.0: 2.6B（直接下载），蒸馏版本 109M（直接下载）

Pangu alpha: 2.6B（直接下载），13B（直接下载），200B（未开放）

CPM 2.0: 11B（需要申请下载）, 198B（需要申请下载）

EVA: 2.6B（需要申请下载）

PLUG: 25B（没有下载）

注意直接下载，并不等于不需要遵守相应协议。

占用显存估计：

2.6B的模型，假设是float32，可能至少需要10GB显存才能完全加载，float16大概需要一半的5GB显存

11B～13B的模型，假设是float16（注意这里是16），可能需要20～25GB显存才能完全加载

### 应用与训练对比：

CPM 1.0 / Pangu alpha ： 基于GPT-3的decoder模型，可以用于直接NLG生成，类似GPT-3的In context zero/few shot learning，可选prompt-tuning

CPM 2.0 / PLUG ： 基于T5的encoder-decoder结构模型（PLUG是估计的），encoder可以单独使用类似BERT，模型需要进行fine-tune或者prompt-tuning，理论上在训练后可以实现几乎任何NLP任务

EVA ： 基于T5的encoder-decoder结构模型，实现对话/闲聊功能，已经针对对话任务进行特殊训练，encoder虽然可以直接用，但是意义不大，可选fine-tune或者prompt-tuning。

### 一些闲话

- 本文没有对比模型，也没有很实际的方法和能力做这件事，本文只是简单的盘点而已，抱歉；
- 除了信息很少的PLUG模型，前面的4个模型都是使用基于jieba的tokenizer，其实文本作者对于这一点是抱有很大的质疑的，因为jieba就可能会有很多问题，比如受限于jieba本身的性能和分词能力，在谷歌都在考虑干脆完全扔掉tokenizer，直接使用字节的现在，如果下一个中文大模型，还是要用jieba，我只能说真的是非常遗憾的事情。以作者本人的工作经验来说，我宁愿牺牲掉一点点性能，也认为更灵活的tokenizer可以让模型在更多的工业级应用上有更好的适用性。（这段提到的不用tokenizer是指谷歌的这篇论文： ByT5: Towards a token-free future with pre-trained byte-to-byte models https://arxiv.org/pdf/2105.13626 ）
- 虽然本文不进行仔细评测，不过作为GPT-3的角度去看待这些模型，会发现中文语料和英文预料表现出来的差异是极大的，一方面估计是数据量的差异，中文预料可能就是差别英文预料相差一个数量级甚至更多。另一方面可能是数据本身质量的因素，毕竟wiki本身就比中文的各种百科在广度和质量好要，更别提还有其他的各种知识信息和网站。喜欢玩游戏尤其是欧美游戏的人肯定知道wikia吧，做投资还有英文能力的人肯定知道investopedia吧，他们真的太喜欢wiki这个东西了，所以积累了大量的知识，这在中文世界中是完全不能比的。

### 应用建议

如果你想试试中文版的GPT-3，想尝试一些in context learning或者zero/few shot learning相关研究，可以从CPM 1.0和Pangu alpha开始尝试

如果你想寻找与测试prompt-tuning相关的方法，或者寻找比mT5更了解中文的模型，请从CPM 2.0开始尝试

如果你想测试当前开放了的最大的中文对话模型，可以从EVA开始尝试
