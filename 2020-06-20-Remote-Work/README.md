# 我在办公室远程办公？四个远程写代码的工具

<!-- toc -->

- [1、基于VS Code与SSH进行远程编程](#1%E5%9F%BA%E4%BA%8Evs-code%E4%B8%8Essh%E8%BF%9B%E8%A1%8C%E8%BF%9C%E7%A8%8B%E7%BC%96%E7%A8%8B)
- [2. 基于浏览器的VS Code（？）](#2-%E5%9F%BA%E4%BA%8E%E6%B5%8F%E8%A7%88%E5%99%A8%E7%9A%84vs-code)
- [3. Jupyter Notebook / Jupyter Lab](#3-jupyter-notebook--jupyter-lab)
- [4. Google Colab / Azure Notebook](#4-google-colab--azure-notebook)

<!-- tocstop -->

今年因为特殊情况，很多公司都开始远程办公，阻碍程序员远程办公的东西有很多，其中一个是如何远程写代码、调试，另一个就是如何远程沟通，这里只讲一些远程写代码的经验。

远程写代码有什么好处呢？

1. 服务器更加安全，可以按照策略配置自动备份等等策略，避免自己在本地作死，电脑坏了、文件删了等等；
2. 随时随地工作，不局限于你在家还是在办公室，也不局限于你本地电脑的性能，反正大部分操作都在远端完成，本地无论是ipad mini还是最高配的游戏本，其实并没有区别；
3. 对于公司，另外还有一些附加好处，我认为未来的发展方向是以后公司都可以考虑给每个写代码的员工配置云编程环境，这样一来可以给每个员工都选购性能不需要太好的电脑，毕竟无论是编程、调试、模型训练理论上都可以在云端进行，也同时一定程度上避免了资料丢失、信息泄漏的风险。

## 1、基于VS Code与SSH进行远程编程

参考微软官方的[介绍文章 《Remote Development using SSH》 https://code.visualstudio.com/docs/remote/ssh](https://code.visualstudio.com/docs/remote/ssh)

首先需要VS Code，其次本地需要有一个支持SSH的客户端，macOS和Linux一般都已经有了，Windows早起版本可以通过安装Git-scm解决，最新版本可以通过[安装OpenSSH Client解决 https://docs.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse](https://docs.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse)。

确认你可以通过无密码的方法（即密钥方式）登录SSH服务器，至于如何配置本文不再阐述，可以随意搜索如“SSH无密码登录”这样的关键词可以得到大量信息。
这一步是否成功可以在命令行下测试。

其次需要支持SSH的服务器，笔者使用的是国内云厂商的服务器资源，包年包月，如果为编程考虑的话优先有几个可以考虑的方向：

1. 内存足够，推荐使用>=4GB的服务器，毕竟在线编程同样也需要一定程度的在线调试，当然如果内存太低做很多事情会很不方便；
2. CPU可以低，2核在大部分时候足够了；
3. 可以考虑一些云厂商的共享资源类，或者CPU使用受限类服务器，因为大部分时候编程需要的是内存足够，而不是CPU足够强劲，所以共享抢占类的服务器只在部分需要调试/编译/安装的少部分时候占用大量CPU，大部分时候普通编程的时候其实不需要占用太多CPU资源，这就很合适了，关键是便宜啊，这类服务器价格可以低至30～200元/月就可以拥有4GB/8GB内存。

最后，在VS Code中，按F1（或command/ctrl + shift + P），打开命令模式，选择“Remote-SSH: Connect to Host”，按照提示输入自己的ssh命令等，即可登录成功，从此开启远程开发之旅。

## 2. 基于浏览器的VS Code（？）

大家要知道，VS Code本身也其实是基于JS/TS开发，运行在Webkit上的桌面程序，类似于使用过Electron之类的程序将web程序封装到本地，所以它当然也可以移植到直接运行在浏览器上。

（iPad mini/iPhone上用vscode！！？？）

这就是另一个很有趣的项目[code-server https://github.com/cdr/code-server](https://github.com/cdr/code-server)

安装也很简单，首先有一个类似上面章节提到的服务器，然后在上面运行code-server的安装命令就可以

```bash
$ url -fsSL https://code-server.dev/install.sh | sh
```

当然也可以参考docker hub中的介绍，[用docker的方式安装code-server https://hub.docker.com/r/linuxserver/code-server ](https://hub.docker.com/r/linuxserver/code-server)

想象一下，如果一个公司把所有开发人员的环境，都用docker部署在一个（或多个）巨大的服务器上，所有人都用浏览器连接属于自己的环境，但是计算资源是共享的，备份/调试/代码安全也都同时可以保证。

甚至以后我们是不是可以把这些浏览器访问作为一种trigger，变成基于某种FaaS的开发环境，使用时才启用，不用时就关闭，不用时没有任何计算资源费用（当然可能有存储资源费用）。

当然上面这两条都不太容易实现，不过我相信未来会逐步这么发展。

## 3. Jupyter Notebook / Jupyter Lab

对于使用Python要进行如数据科学/机器学习等方面工作的人，这是经常使用的环境，而大家也应该知道，它天然就是基于浏览器在运行的。

Jupyter Lab是一个Notebook的扩展，可以在服务器上使用账号控制的方法完成更多的登录、管理等操作。

这里简单介绍一下Notebook的配置。

Jupyter notebook默认只考虑了本地的情况，所以没有配置密码，只开启了验证token，我们要在服务器上运行自然不能这么随意，至少也要配置密码。

```bash
# 首先安装jupyter notebook
$ pip install jupyter notebook
# 生成jupyter配置文件
$ jupyter notebook --generate-config
# 修改jupyter配置文件
$ vim ~/.jupyter/jupyter_notebook_config.py
```

在配置文件中可能需要配置，加到末尾：
```python
# 不自动中启动时打开浏览器
c.NotebookApp.open_browser = False
# 绑定ip
c.NotebookApp.ip = '*'
# 绑定端口
c.NotebookApp.port = 8888
```

最后用`jupyter notebook password`命令配置密码完成。

熟练使用jupyter也可以用来开发简单的web应用，有服务器的话这些应用更是可以直接运行在云端，参考之前写的文章[将Jupyter Notebook变成Web APP：Voila](https://zhuanlan.zhihu.com/p/127300044)

## 4. Google Colab / Azure Notebook

Google Colab和Azure Notebook都可以认为是基于jupyter notebook的一种变种，主要缺点是因为特殊情况，不好访问。

但是它们都会带来巨大的优势。

举个例子，Colab中是提供GPU服务的，而且它的网络速度可快得多，也就是很方便的可以进行各种大数据集的研究，kaggle的实验等等。

当然默认Colab分配的GPU一般是K40，不过如果你购买了[Colab Pro服务](https://colab.research.google.com/signup)，也就是每月9.99美元，几乎可以保证每次分配到P100的GPU，这个GPU是一个什么水平呢，是在GTX 1080的水平的，而配置一台包含1080的机器的折算年费，肯定要比9.99美元/月要多。如果你的模型可以通过TPU运行，那效果则更好。

当然Colab在你不使用，无浏览器动作等等之后，会自主收回运行环境，但是只要我们记得保存中间结果在Google Drive，重新打开继续运行也是一样的，虽然麻烦一点，不过性价比依然很高。

[Azure Notebook](http://notebooks.azure.com/)也提供了免费的运行服务器，不过相比Colab稍有逊色，当然我也更希望它们这个服务能在国内的Azure上提供服务，这样就很好了。

实际上国内的很多厂商也开始提供类似在线notebook的服务，包括百度云/华为云，实话说，它们都不如Google大方，如果可能的话，还是优先考虑/使用一下Colab，体验世界一流的资源/服务是什么样的。

---

本文就是使用VS Code连接到远程服务器上，用Markdown书写的初稿，习惯就会慢慢成为自然。
