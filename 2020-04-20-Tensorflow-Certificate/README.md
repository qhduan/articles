# TensorFlow认证考试简介

今年的TensorFlow开发者大会，谷歌宣布了一个名为“TensorFlow Developer Certificate program”的项目/考试。

笔者对这个考试最大的好奇是：考什么，如何考。

这两方面，考什么，类似于谷歌认为什么才能算是合格的TensorFlow开发者的标准。如何考，是作为一个完全远程的考试，怎么考，如何判定结果。

## 考什么

先给出官方的Handbook （[https://www.tensorflow.org/site-assets/downloads/marketing/cert/TF_Certificate_Candidate_Handbook.pdf](https://www.tensorflow.org/site-assets/downloads/marketing/cert/TF_Certificate_Candidate_Handbook.pdf)）

关于考什么，官方的Handbook说的相当详细了，在Handbook里面有一个最长的章节，即“Skills checklist”，这里面也就相对的列出了谷歌认为什么样的技能拥有了，算是符合一个合格的TensorFlow开发者，其中大部分其中的描述也适用于其他机器学习框架。

### Build and train neural network models using TensorFlow 2.x

在checklist的第一部分，首先明确了这是针对TensorFlow 2.x的认证，2.x现在的逐渐发展是向着Keras，或者说pyTorch那样的模块化走的。

- Use TensorFlow 2.x.
- Build, compile and train machine learning (ML) models using TensorFlow.
- Preprocess data to get it ready for use in a model.
- Use models to predict results.
- Build sequential models with multiple layers.
- Build and train models for binary classification.
- Build and train models for multi-class categorization.
- Plot loss and accuracy of a trained model.
- Identify strategies to prevent overfitting, including augmentation and dropout.
- Use pretrained models (transfer learning).
- Extract features from pre-trained models.
- Ensure that inputs to a model are in the correct shape.
- Ensure that you can match test data to the input shape of a neural network.
- Ensure you can match output data of a neural network to specified input shape for test data.
- Understand batch loading of data.
- Use callbacks to trigger the end of training cycles.
- Use datasets from different sources.
- Use datasets in different formats, including json and csv.
- Use datasets from tf.data.datasets.

其实机器学习考试是很难考核模型本身“效果”的优劣的，例如某个模型必须准确率到99.5%之类，毕竟有一定随机性的，所以其实重要的是，至少能把符合输入输出要求的模型搭建起来，也就是对于问题的基本建模能力。

第一部分上面这些条目也主要是如何搭建模型，尤其是如何使用TensorFlow 2.x自己的组件，如tf.data去搭建模型的输入输出。

### 其他部分

#### 图像

- Define Convolutional neural networks with Conv2D and pooling layers.
- Build and train models to process real-world image datasets.
- Understand how to use convolutions to improve your neural network.
- Use real-world images in different shapes and sizes..
- Use image augmentation to prevent overfitting.
- Use ImageDataGenerator.
- Understand how ImageDataGenerator labels images based on the directory structure.

#### 文本

- Build natural language processing systems using TensorFlow.
- Prepare text to use in TensorFlow models.
- Build models that identify the category of a piece of text using binary categorization
- Build models that identify the category of a piece of text using multi-class categorization
- Use word embeddings in your TensorFlow model.
- Use LSTMs in your model to classify text for either binary or multi-class categorization.
- Add RNN and GRU layers to your model.
- Use RNNS, LSTMs, GRUs and CNNs in models that work with text.
- Train LSTMs on existing text to generate text (such as songs and poetry)

#### 时间序列

- Train, tune and use time series, sequence and prediction models.
- Prepare data for time series learning.
- Understand Mean Average Error (MAE) and how it can be used to evaluate accuracy of sequence models.
- Use RNNs and CNNs for time series, sequence and forecasting models.
- Identify when to use trailing versus centred windows.
- Use TensorFlow for forecasting.
- Prepare features and labels.
- Identify and compensate for sequence bias.
- Adjust the learning rate dynamically in time series, sequence and prediction models.

上面提到的任务、方法涉及到：图像卷积、图像分类、图像增强、文本二分类、文本多分类、RNN的运用、文本生成、时间序列预测这些任务。

也就是说，要通过考试，应该对上面的任务有一定的理解，知道相关问题如何建模，知道如何用TensorFlow 2.x自有的方法处理或建模。

## 如何考

这也是一个很有意思一点，就是这次考试是用PyCharm安装一个考试的插件，连接远程的服务器进行考试，在连接成功那一刻开始，考试就会开始计时，在这个过程中可以任意提交模型/完成所有题目的答卷。

远程考试，事实上就是一种开卷考试了，考核的基本上是建模和基本解决问题的能力，而不是得到最优解的能力。所以就算通过了这个认证，其实也就是说你有用TensorFlow解决一些问题的基本能力，不过是否有很好的优化模型、创造模型、应对更现实多变的情况的建模能力，就不在这个考试范畴内。

引申一下，这种考试系统感觉上还是比较好的，可以一定程度用在远程面试、内部考核上，不一定非要PyCharm，可以考虑docker + code-server就可以实现类似的功能，最后由一个统一的认真服务器测试输出的模型，甚至是程序源代码即可。（code-server：web版的vs code）

## FAQ

- 国内考试需要护照，和“能上网”的PyCharm
- 考试只考TensorFlow 2.x，或者你可以粗暴理解为只考Keras相关也没问题
- 提交只提交save的模型
- 考试不需要显卡或太好的电脑，如果这个考试里面出现了必须用2080显卡训练2个小时才能实现，那是不是太过分了，所以如果你在过程中遇到了这个问题，那大概率说明你实现错了
- 5个模型
- 具体如何判断是否通过未知，而且规则也可能改吧
- 考试的时候可以查阅资料的范围，参考官方说法：You may use whatever learning resources you would normally use during your ML development work.
