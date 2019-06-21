# Darknet源码解析
Darknet是一个较为轻型的完全基于C与CUDA的开源深度学习框架，其主要特点就是容易安装，没有任何依赖项（OpenCV都可以不用），移植性非常好，支持CPU与GPU两种计算方式。

更多信息（包括安装、使用）可以参考：[Darknet: Open Source Neural Networks in C](https://pjreddie.com/darknet/)

# 为什么要做这个？

我在阅读Yolo论文的时候对很多实现细节不太明白，但是所幸我找到了讲解Yolo源码的一些博客，工程，给了我很大的启发，所以我打算在他们的基础上从宏观和微观的角度来理解一下DarkNet并将笔记记录到这个工程里。

## 宏观方面

- YOLO源码分析1之训练.md
- YOLO源码分析2之网络架构分析.md
- YOLO源码分析3之前向传播.md
- YOLO源码分析4之反向传播.md
- YOLO源码分析5之grid cell和bbox理解.md

## 微观方面

- 我会对src里面部分文件进行详细的注释，并将注释后的文件名更新在此处。



# 未理解的问题

我会将没理解的问题记录在problem.md中，欢迎大家一起讨论和解决。

# 参考博客及工程

https://github.com/hgpvision/darknet

https://blog.csdn.net/u014540717/column/info/13752

https://blog.csdn.net/gzj2013/article/details/82456548