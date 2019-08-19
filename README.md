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

读取配置文件由src/parser.c中的`read_cfg()`函数实现：

```c++
/*
 * 读取神经网络结构配置文件（.cfg文件）中的配置数据， 将每个神经网络层参数读取到每个
 * section 结构体 (每个 section 是 sections 的一个节点) 中， 而后全部插入到
 * list 结构体 sections 中并返回
 * 
 * \param: filename    C 风格字符数组， 神经网络结构配置文件路径
 * 
 * \return: list 结构体指针，包含从神经网络结构配置文件中读入的所有神经网络层的参数
 * 每个 section 的所在行的开头是 ‘[’ , ‘\0’ , ‘#’ 和 ‘;’ 符号开头的行为无效行, 除此
 *之外的行为 section 对应的参数行. 每一行都是一个等式, 类似键值对的形式.

 *可以看到, 如果某一行开头是符号 ‘[’ , 说明读到了一个新的 section: current, 然后第945行
 *list_insert(options, current);` 将该新的 section 保存起来.

 *在读取到下一个开头符号为 ‘[’ 的行之前的所有行都是该 section 的参数, 在第 957 行 
 *read_option(line, current->options) 将读取到的参数保存在 current 变量的 options 中. 
 *注意, 这里保存在 options 节点中的数据为 kvp 键值对类型.

 *当然对于 kvp 类型的参数, 需要先将每一行中对应的键和值(用 ‘=’ 分割) 分离出来, 然后再
 *构造一个 kvp 类型的变量作为节点元素的数据.
 */

list *read_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    //一个section表示配置文件中的一个字段，也就是网络结构中的一层
    //因此，一个section将读取并存储某一层的参数以及该层的type
    if(file == 0) file_error(filename);
    char *line;
    int nu = 0; //当前读取行号
    list *options = make_list(); //options包含所有的神经网络层参数
    section *current = 0; //当前读取到某一层
    while((line=fgetl(file)) != 0){ 
        ++ nu;
        strip(line); //去除读入行中含有的空格符
        switch(line[0]){
            // 以 '[' 开头的行是一个新的 section , 其内容是层的 type 
            // 比如 [net], [maxpool], [convolutional] ...
            case '[':
                current = malloc(sizeof(section));
                list_insert(options, current);
                current->options = make_list();
                current->type = line;
                break;
            case '\0': //空行
            case '#':  //注释
            case ';': //空行
                free(line); // 对于上述三种情况直接释放内存即可
                break;
            default:
                // 剩下的才真正是网络结构的数据，调用 read_option() 函数读取
                // 返回 0 说明文件中的数据格式有问题，将会提示错误
                if(!read_option(line, current->options)){
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file);
    return options;
}
```

### 链表的插入操作

保存 section 和每个参数组成的键值对时使用的是 list_insert() 函数, 前面提到了参数保存的结构其实是大链( 节点为 section )上边挂着很多小链( 每个 section 节点的各个参数)。`list_insert()`函数实现了链表插入操作，该函数定义在src/list.c 文件中：

```c++
/*
 * \brief: 将 val 指针插入 list 结构体 l 中，这里相当于是用 C 实现了 C++ 中的 
 *         list 的元素插入功能
 * 
 * \prama: l    链表指针
 *         val  链表节点的元素值
 * 
 * 流程： list 中保存的是 node 指针. 因此，需要用 node 结构体将 val 包裹起来后才可以
 *       插入 list 指针 l 中
 * 
 * 注意: 此函数类似 C++ 的 insert() 插入方式；
 *      而 opion_insert() 函数类似 C++ map 的按值插入方式，比如 map[key]= value
 *      
 *      两个函数操作对象都是 list 变量， 只是操作方式略有不同。
*/
void list_insert(list *l, void *val)
{
	node *new = malloc(sizeof(node));
	new->val = val;
	new->next = 0;
    // 如果 list 的 back 成员为空(初始化为 0), 说明 l 到目前为止，还没有存入数据  
    // 另外, 令 l 的 front 为 new （此后 front 将不会再变，除非删除） 
	if(!l->back){
		l->front = new;
		new->prev = 0;
	}else{
		l->back->next = new;
		new->prev = l->back;
	}
	l->back = new;
	++l->size;
}
```

可以看到, 插入的数据都会被重新包装在一个新的 node : 变量 new 中, 然后再将这个节点插入到链表中。网络结构解析到链表中后还不能直接使用, 如果仅仅想使用某一个参数而不得不每次都遍历整个链表, 这样就会导致程序效率变低, 最好的办法是将其保存到一个结构体变量中, 使用的时候按照成员进行访问。

### 将链表中的网络结构保存到network结构体

- 首先来看看network结构体的定义，在include/darknet.h中：

  ```c++
  //定义network结构
  typedef struct network{
      int n; //网络的层数，调用make_network(int n)时赋值
      int batch; //一批训练中的图片参数，和subdivsions参数相关
      size_t *seen; //目前已经读入的图片张数(网络已经处理的图片张数) 
      int *t;
      float epoch; //到目前为止训练了整个数据集的次数
      int subdivisions;
      layer *layers;  //存储网络中的所有层  
      float *output;
      learning_rate_policy policy; // 学习率下降策略: TODO
      // 梯度下降法相关参数  
      float learning_rate;
      float momentum;
      float decay;
      float gamma;
      float scale;
      float power;
      int time_steps;
      int step;
      int max_batches;
      float *scales;
      int   *steps;
      int num_steps;
      int burn_in;
  
      int adam;
      float B1;
      float B2;
      float eps;
  
      int inputs;
      int outputs;
      int truths;
      int notruth;
      int h, w, c;
      int max_crop;
      int min_crop;
      float max_ratio;
      float min_ratio;
      int center;
      float angle;
      float aspect;
      float exposure;
      float saturation;
      float hue;
      int random;
      //darknet 为每个 GPU 维护一个相同的 network, 每个 network 以 gpu_index 区分
      int gpu_index;
      tree *hierarchy;
      //中间变量，用来暂存某层网络的输入（包含一个 batch 的输入，比如某层网络完成前向，
      //将其输出赋给该变量，作为下一层的输入，可以参看 network.c 中的forward_network() 
      //与 backward_network() 两个函数 ）
      float *input;
      // 中间变量，与上面的 input 对应，用来暂存 input 数据对应的标签数据（真实数据）
      float *truth;
      // 中间变量，用来暂存某层网络的敏感度图（反向传播处理当前层时，用来存储上一层的敏
      //感度图，因为当前层会计算部分上一层的敏感度图，可以参看 network.c 中的 backward_network() 函数） 
      float *delta;
      // 网络的工作空间, 指的是所有层中占用运算空间最大的那个层的 workspace_size, 
      // 因为实际上在 GPU 或 CPU 中某个时刻只有一个层在做前向或反向运算
      float *workspace;
       // 网络是否处于训练阶段的标志参数，如果是则值为1. 这个参数一般用于训练与测试阶段有不
      // 同操作的情况，比如 dropout 层，在训练阶段才需要进行 forward_dropout_layer()
      // 函数， 测试阶段则不需要进入到该函数
      int train;
      int index;// 标志参数，当前网络的活跃层 
      float *cost;
      float clip;
  
  #ifdef GPU
      float *input_gpu;
      float *truth_gpu;
      float *delta_gpu;
      float *output_gpu;
  #endif
  
  } network;
  ```

- 为网络结构体分配内存空间，函数定义在src/network.c文件中：

  ```c++
  //为网络结构体分配内存空间
  network *make_network(int n)
  {
      network *net = calloc(1, sizeof(network));
      net->n = n;
      net->layers = calloc(net->n, sizeof(layer));
      net->seen = calloc(1, sizeof(size_t));
      net->t    = calloc(1, sizeof(int));
      net->cost = calloc(1, sizeof(float));
      return net;
  }
  ```

  在src/parser.c中的`parse_network_cfg()`函数中，从net变量开始，依次为其中的指针变量分配内存。由于第一个段`[net]`中存放的是和网络并不直接相关的配置参数, 因此网络中层的数目为 sections->size - 1，即：`network *net = make_network(sections->size - 1);`

- 将链表中的网络参数解析后保存到 network 结构体，配置文件的第一个段一定是`[net]`段，该段的参数解析由`parse_net_options()`函数完成，函数定义在src/parser.c中。之后的各段都是网络中的层。比如完成特定特征提取的卷积层，用来降低训练误差的shortcur层和防止过拟合的dropout层等。这些层都有特定的解析函数：比如`parse_convolutional()`, `parse_shortcut()`和`parse_dropout()`。每个解析函数返回一个填充好的层l，将这些层全部添加到network结构体的layers数组中。即是：`net->layers[count] = l;`另外需要注意的是这行代码：`if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;`，其中workspace代表网络的工作空间，指的是所有层中占用运算空间最大那个层的workspace。因为在CPU或GPU中某个时刻只有一个层在做前向或反向传播。 输出层只能在网络搭建完毕之后才可以确定，输入层需要考虑`batch_size`的因素，truth是输入标签，同样需要考虑`batch_size`的因素。

  ```c++
  	layer out = get_network_output_layer(net);
      net->outputs = out.outputs;
      net->truths = out.outputs;
      if(net->layers[net->n-1].truths) net->truths = net->layers[net->n-1].truths;
      net->output = out.output;
      net->input = calloc(net->inputs*net->batch, sizeof(float));
      net->truth = calloc(net->truths*net->batch, sizeof(float));
  ```

- 到这里，网络的宏观解析结束。`parse_network_cfg()`(src/parser.c中)函数返回解析好的network类型的指针变量。

### 为啥需要中间数据结构缓存？

这里可能有个疑问，为什么不将配置文件读取并解析到 network 结构体变量中, 而要使用一个中间数据结构来缓存读取到的文件呢？如果不使用中间数据结构来缓存. 将读取和解析流程串行进行的话, 如果配置文件较为复杂, 就会长时间使文件处于打开状态。 如果此时用户更改了配置文件中的一些条目, 就会导致读取和解析过程出现问题。分开两步进行可以先快速读取文件信息到内存中组织好的结构中, 这时就可以关闭文件. 然后再慢慢的解析参数。这种机制类似于操作系统中断的底半部机制, 先处理重要的中断信号, 然后在系统负荷较小时再处理中断信号中携带的任务。

## 加载训练样本数据

darknet的数据加载在src/data.c中实现，`load_data()`函数调用流程如下：`load_data(args)->load_threads()->load_data_in_threads()->load_thread()->load_data_detection()`，前四个函数都是在对线程的调用进行封装，主要是个线程的加载任务量。最底层的数据加载任务由 `load_data_detection()` 函数完成。所有的数据(图片数据和标注信息数据)加载完成之后再拼接到一个大的数组中。在darknet中，图片的存储形式是一个行向量，向量长度为`h*w*3`。同时图片被归一化到[0, 1]之间。





# 参考资料

https://blog.csdn.net/gzj2013/article/details/84837198

https://blog.csdn.net/u014540717/column/info/13752

https://github.com/hgpvision/darknet