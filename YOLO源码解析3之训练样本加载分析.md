# 训练样本加载

这篇笔记主要分析data.c文件，load_data()函数的调用流程如下:

```c++
load_data(args) -> load_threads() -> load_data_in_thread() -> 
load_thread() -> load_data_detection()
```

前面的四个函数都是在对线程的调用进行封装, 主要是分配每个线程的加载任务量. 最底层的数据加载任务由 load_data_detection() 函数完成.  所有的数据(图片数据和标注信息数据)加载完成之后再拼接到一个大的数组中。在 darknet 中, 图片的存储形式是一个行向量, 向量长度为[batch, h, w, 3]. 同时图片数据被归一化到 0-1 之间。