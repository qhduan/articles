# NLP在TensorFlow 2.x中的最佳实战

NLP Best Practice with TensorFlow

## TextVectorization

这个特性的介绍，什么时候用，怎么用

## tf.strings

tf.strings的作用，TextVectorization我们怎么自己实现

## bucket_by_sequence_length

问题是什么？

因为句子长度不一致

不一致会导致什么问题？

浪费

那么怎么解决？

分bucket

TensorFlow中的方法？

用bucket_by_sequence_length

## Bert in TensorFlow 2.x

## Gradients Clip

从经验上来看，NLP中很容易出现梯度爆炸的问题，尤其是像Albert这样的重复利用参数的情况。

个人总体估计有两个主要原因：

一个是NLP的参数是人为训练的而不是天然形成的，大家知道所谓的embedding方法其实是我们给每个词，或者说token，认为设定一个embedding，而不是像图片、声音那样有天然的embedding，因为这个embedding是人为得到的，所以也会训练，而这种训练结果的累计可能使embedding逐渐不稳定；

其次一个原因可能是NLP竟然面对大量的softmax的情况，例如一次序列标注任务，就是tagging任务，一个句子可能上百个词的标签，每个都是一个softmax，而一个批次可能几十个或上百个这样的句子，就是成千个softmax。
softmax相比sigmoid等分布一个优点就是，梯度回传比较大，模型更好训练。
而在这里它也是一个缺点，就是回传梯度比较大，所以比较容易导致梯度爆炸。

解决梯度爆炸的直观表现，就是你发现训练中损失函数值一般是在震荡中下降的，但是你发现它竟然逐渐上涨，活着直接出现NaN的值。

解决的最简单的方法，一个是降低学习率，一个是增加梯度剪裁。

降低学习率当然是一个可选的方法，不过它主要的问题是同时也降低了整体的学习速度。

如果产生梯度爆炸的样本是比较多数，例如你发现无论怎么随机打乱样本，总是在训练几步之后都会梯度爆炸，这个就很可能是学习率太高了。

如果这样的情况不常发上，那更可能是少数的几个样本出错了，这个时候就需要考虑梯度裁剪。

梯度裁剪就是使用clip norm和clip value两个函数，分别裁剪梯度的norm或绝对值。

这两个可以都用，也可以只用一个。

个人经验来说，可以考虑只用clip value，因为用clip norm的话可能影响总体的学习速度。
