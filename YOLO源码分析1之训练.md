# DarkNet配置及YOLO训练

关于DarkNet的详细配置和制作VOC2007数据集训练YOLOV1/V2/V3请看我的博客链接：https://blog.csdn.net/just_sort/article/details/81389571

# 源码解析

1.首先我们看一下训练的数据流，从main函数开始看，该函数在`examples/darknet.c`中：

![](images/train.jpg)

2. 接下来我们跟进yolo.c去看看：

```c++
void run_yolo(int argc, char **argv)
{
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .2);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    int avg = find_int_arg(argc, argv, "-avg", 1);
    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5]: 0;
    if(0==strcmp(argv[2], "test")) test_yolo(cfg, weights, filename, thresh);
    //第二个参数是train，所以执行train_yolo函数
    else if(0==strcmp(argv[2], "train")) train_yolo(cfg, weights);
    else if(0==strcmp(argv[2], "valid")) validate_yolo(cfg, weights);
    else if(0==strcmp(argv[2], "recall")) validate_yolo_recall(cfg, weights);
    else if(0==strcmp(argv[2], "demo")) demo(cfg, weights, thresh, cam_index, filename, voc_names, 20, frame_skip, prefix, avg, .5, 0,0,0,0);
}

void train_yolo(char *cfgfile, char *weightfile)
{
    //train_images是要训练的图像所在位置
    char *train_images = "/data/voc/train.txt";
    //训练完模型保存的位置
    char *backup_directory = "/home/pjreddie/backup/";
   /*srand函数是随机数发生器的初始化函数。
    srand和rand()配合使用产生伪随机数序列。rand函数在产生随机数前，需要系统提供的生成伪随机数序列的
    种子，rand根据这个种子的值产生一系列随机数。如果系统提供的种子没有变化，每次调用rand函数生成的伪
    随机数序列都是一样的。*/
    srand(time(0));
    //用于从文件全路径字符串中提取出主要信息，比如从cfg.yolo.cfg中提取出yolo
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    //平均损失，初始化为-1
    float avg_loss = -1;
    //解析网络结构
    network *net = load_network(cfgfile, weightfile, 0);
    //打印学习率，动量，学习率衰减参数
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    //imgs是一次加载到内存的图像数量，如果占内存太大的话可以把subdivisions或者batch调小一点
    int imgs = net->batch*net->subdivisions;
    //net.seen就是网络已经训练的图片数量。算出的i就是已经经过了多少次训练
    int i = *net->seen/imgs;
    data train, buffer;


    layer l = net->layers[net->n - 1];
    //side就是yoloV1论文中的grid cell即7
    int side = l.side;
    //分类数，对于VOC就是21
    int classes = l.classes;
    //[非均衡数据集处理：利用抖动(jittering)生成额外数据](http://weibo.com/1402400261/EgMr4vCC2?type=comment#_rnd1478833653326)
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.paths = paths;
    //一次加载到内存中的图片数量
    args.n = imgs;
    //plist的size等于待训练的图片总数
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    //7*7个grid cell
    args.num_boxes = side;
    args.d = &buffer;
    args.type = REGION_DATA;
    //调节图片旋转角度、曝光度、饱和度、色调等，来做数据增强
    args.angle = net->angle;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    //声明线程id
    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;
    //这里使用线程阻塞的方式来加载一个batch的数据
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net->max_batches){
        i += 1;
        time=clock();
         /*pthread_join()函数，以阻塞的方式等待thread指定的线程结束。当函数返回时，被等待线程的资源被
        收回。如果线程已经结束，那么该函数会立即返回。*/
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_in_thread(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        time=clock();
        //开始训练
        float loss = train_network(net, train);
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        //保存模型和打印一些训练中的信息
        printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
        if(i%1000==0 || (i < 1000 && i%100 == 0)){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}
```