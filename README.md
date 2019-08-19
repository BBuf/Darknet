# Darknet源码阅读
Darknet是一个较为轻型的完全基于C与CUDA的开源深度学习框架，其主要特点就是容易安装，没有任何依赖项（OpenCV都可以不用），移植性非常好，支持CPU与GPU两种计算方式。

更多信息（包括安装、使用）可以参考：[Darknet: Open Source Neural Networks in C](https://pjreddie.com/darknet/)

# 为什么要做这个？

我在阅读Yolo论文的时候对很多实现细节不太明白，但是所幸我找到了讲解Yolo源码的一些博客，工程，给了我很大的启发，所以我打算在他们的基础上从宏观和微观的角度来理解一下DarkNet并将笔记记录到这个工程里。

# DarkNet源码阅读之主线

## 关于分析主线的确定

darknet相比当前训练的c/c++主流框架来讲，具有编译速度快，依赖少，易部署等众多优点，我们先定位到examples/darknet.c的main函数，这是这个框架实现分类，定位，回归，分割等功能的初始入口。

![](image/1.png)

## 目标检测-run_detector

我们这里主要来分析一下目标检测，也就是examples/detector.c中的run_detector函数。可以看到这个函数主要包括了训练-测试-验证三个阶段。

![](image/2.png)

## 训练检测器-train_detector

由于训练，验证和测试阶段代码几乎是差不多的，只不过训练多了一个反向传播的过程。所以我们主要分析一下训练过程，训练过程是一个比较复杂的过程，不过宏观上大致分为解析网络配置文件，加载训练样本图像和labels，开启训练，结束训练保存模型这样一个过程，整体代码如下：

![](image/3.png)

## 解析配置文件(.cfg)

### 配置文件长啥样？

![](image/4.png)

可以看到配置参数大概分为2类：

- 与训练相关的项，以 [net] 行开头的段. 其中包含的参数有: `batch_size, width,height,channel,momentum,decay,angle,saturation, exposure,hue,learning_rate,burn_in,max_batches,policy,steps,scales`。
- 不同类型的层的配置参数. 如` [convolutional], [short_cut], [yolo], [route], [upsample] `层等。

在src/parse.c中我们会看到一行代码，`net->batch /= net->subdivisions;`，也就是说`batch_size` 在 darknet 内部又被均分为 `net->subdivisions`份, 成为更小的`batch_size`。 但是这些小的 `batch_size` 最终又被汇总, 因此 darknet 中的`batch_size = net->batch / net->subdivisions * net->subdivisions`。此外，和这个参数相关的计算训练图片数目的时候是这样，`int imgs = net->batch * net->subdivisions * ngpus;`，这样可以保证`imgs`可以被`subdivisions`整除，因此，通常将这个参数设为8的倍数。从这里也可以看出每个gpu或者cpu都会训练`batch`个样本。

我们知道了参数是什么样子，那么darknet是如何保存这些参数的呢？这就要看下基本数据结构了。

### 基本数据结构？

Darknet是一个C语言实现的神经网络框架，这就决定了其中大多数保存数据的数据结构都会使用链表这种简单高效的数据结构。为了解析网络配置参数, darknet 中定义了三个关键的数据结构类型。list 类型变量保存所有的网络参数, section类型变量保存的是网络中每一层的网络类型和参数, 其中的参数又是使用list类型来表示.  kvp键值对类型用来保存解析后的参数变量和参数值。

- list类型定义在include/darknet.h文件中，代码如下：

  ```c++
  //链表上的节点
  typedef struct node{
      void *val;
      struct node *next;
      struct node *prev;
  } node;
  
  //双向链表
  typedef struct list{
      int size; //list的所有节点个数
      node *front; //list的首节点
      node *back; //list的普通节点
  } list;
  
  ```

- section 类型定义在src/parser.c文件中，代码如下：

  ```c++
  typedef struct{
      char *type;
      list *options;
  }section;
  ```

- kvp 键值对类型定义在src/option_list.h文件中，具体定义如下：

  ```c++
  typedef struct{
      char *key;
      char *val;
      int used;
  } kvp;
  ```

在darknet的网络配置文件(.cfg结尾)中，以`[`开头的行被称为一个段(section)。所有的网络配置参数保存在list类型变量中，list中有很多的sections节点，每个sections中又有一个保存层参数的小list，整体上出现了一种大链挂小链的结构。大链的每个节点为section，每个section中包含的参数保存在小链中，小链的节点值的数据类型为kvp键值对，这里有个图片可以解释这种结构。

![](image/5.jpg)

我们来大概解释下该参数网，首先创建一个list，取名sections，记录一共有多少个section（一个section存储了CNN一层所需参数）；然后创建一个node，该node的void类型的指针指向一个新创建的section；该section的char类型指针指向.cfg文件中的某一行（line），然后将该section的list指针指向一个新创建的node，该node的void指针指向一个kvp结构体，kvp结构体中的key就是.cfg文件中的关键字（如：`batch，subdivisions`等），val就是对应的值；如此循环就形成了上述的参数网络图。

### 解析并保存网络参数到链表中



# 参考资料

https://blog.csdn.net/gzj2013/article/details/84837198

https://blog.csdn.net/u014540717/column/info/13752

https://github.com/hgpvision/darknet