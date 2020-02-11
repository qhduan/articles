# 自然语言理解的应用 SNIPS-NLU

    AI工程：尽量不自己造轮子

## 什么是自然语言理解

自然语言理解，即Natural Language Understanding，可以认为是特指对话系统/对话机器人/Chatbot中的NLU组件/模块，也可以认为是所有自然语言理解类任务的总称。

它的目的，是希望计算机能理解人类语言、自然语言，当然也可以是非自然语言，如命令式语言的解析。

它的作用本质上是希望将输入的语言符号，例如自然语言句子、段落，理解为逻辑符号、逻辑推理、变量、实体或者任何可推理可判断的东西。

## 问答的例子

在问答或搜索中，我们可以假设为用户提问句子与问答系统，或搜索引擎中被索引的句子的一个匹配过程，并且应该把与用户问题更匹配的句子给出，或至少排序在前面。

例如用户提问的句子是：“公司A在2018年之后的收入”

假设系统内索引的句子有两个：

- 公司A在2016年之前的收入是……
- 公司A在2020年之前的收入是……

如果只考虑句子的matching问题，那么以上两句话和用户输入的匹配度其实是完全一样的，因为它们的词汇不同点在于2016和2019，而这两个词都在用户搜索句子中未出现，所以并不影响句子的词汇匹配。

但是推理来看，显然2020年之前，应比2016年之前的更符合结果。因为2020年之前肯定包含2018年之后，但是2016年之前显然不会包括2018年之后。也就是说第二句话显然按照人类逻辑应该优先返回。

那么问题就是，在只用搜索引擎技术，或者说只用字符相似度的情况下，是无法做到这样的排序的。

## 用NLU改善问答的例子

以上的例子，显然如果我们能够通过自然语言理解，获取用户搜索的句子的时间，还有计算我们数据库中索引了的数据条目的时间，进行时间重合度/匹配度的计算，就能更好的分清楚到底哪个句子应该排在前面（即更匹配用户意图）。

我们这里尝试用[Snips-nlu](https://github.com/snipsco/snips-nlu)来处理英文时间识别问题

（因为没有好的中文处理工具）

- 公司A在2018年之后的收入
  - The company's revenue after 2018

- 公司A在2016年之前的收入是……
  - The company's revenue before 2016 is xxx
- 公司A在2020年之前的收入是……
  - The company's revenue before 2020 is xxx

以上的句子用如下代码解析：

```python
import json
from snips_nlu_parsers import BuiltinEntityParser
parser = BuiltinEntityParser.build(language="en")
print(json.dumps(parser.parse("The company's revenue after 2018"), indent=4))
print(json.dumps(parser.parse("The company's revenue before 2020 is xxx"), indent=4))
print(json.dumps(parser.parse("The company's revenue before 2016 is xxx"), indent=4))
```

它会分别返回：

```json
[
    {
        "value": "after 2018",
        "range": {
            "start": 22,
            "end": 32
        },
        "entity": {
            "kind": "TimeInterval",
            "from": "2018-01-01 00:00:00 +08:00",
            "to": null
        },
        "alternatives": [],
        "entity_kind": "snips/datePeriod"
    }
]
```

```json
[
    {
        "value": "before 2020",
        "range": {
            "start": 22,
            "end": 33
        },
        "entity": {
            "kind": "TimeInterval",
            "from": null,
            "to": "2020-01-01 00:00:00 +08:00"
        },
        "alternatives": [],
        "entity_kind": "snips/datePeriod"
    }
]
```

```json
[
    {
        "value": "before 2016",
        "range": {
            "start": 22,
            "end": 33
        },
        "entity": {
            "kind": "TimeInterval",
            "from": null,
            "to": "2016-01-01 00:00:00 +08:00"
        },
        "alternatives": [],
        "entity_kind": "snips/datePeriod"
    }
]
```

我们只要把"after 2018"的时间区间结果，分别和"before 2020"与"before 2016"对比，就能得到哪个更符合用户题意了。

## 工程

以上的例子过于简单粗暴，实际情况肯定更复杂，例如这种分析的成本可能过于高了。

还有例如一种情况，就是用户输入的句子无法被snips-nlu识别，

例如用户如果输入的是“Company A's revenue in recent 3 months”

它会被错误的识别为：

```json
[
    {
        "value": "3 months",
        "range": {
            "start": 30,
            "end": 38
        },
        "entity": {
            "kind": "Duration",
            "years": 0,
            "quarters": 0,
            "months": 3,
            "weeks": 0,
            "days": 0,
            "hours": 0,
            "minutes": 0,
            "seconds": 0,
            "precision": "Exact"
        },
        "alternatives": [],
        "entity_kind": "snips/duration"
    }
]
```

用户本意是想说最近3个月公司的收入，但是找到的却是3个月作为一个持续番位，显然不符合结果。

那么在AI工程上往往有两个方法：

1. 重新训练NLU，增加例子，训练一个自己的NLU模型，解决问题
2. 在自然语言之前加规则层，绕开问题

在第二个方法中，例如上面的问题，我们可以写一个规则模板、正则表达式，把“recent 3 month”改为“last 3 month”，这个正则表达式规则相对很好写，然后识别就正确了：

```json
[
    {
        "value": "in last 3 months",
        "range": {
            "start": 20,
            "end": 36
        },
        "entity": {
            "kind": "TimeInterval",
            "from": "2019-11-01 00:00:00 +08:00",
            "to": "2020-02-01 00:00:00 +08:00"
        },
        "alternatives": [],
        "entity_kind": "snips/datePeriod"
    }
]
```

