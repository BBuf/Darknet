#ifndef DARKNET_API
#define DARKNET_API

#if defined(_MSC_VER) && _MSC_VER < 1900
#define inline __inline
#endif

#if defined(DEBUG) && !defined(_CRTDBG_MAP_ALLOC)
#define _CRTDBG_MAP_ALLOC
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <pthread.h>

#ifndef LIB_API
#ifdef LIB_EXPORTS
#if defined(_MSC_VER)
#define LIB_API __declspec(dllexport)
#else
#define LIB_API __attribute__((visibility("default")))
#endif
#else
#if defined(_MSC_VER)
#define LIB_API
#else
#define LIB_API
#endif
#endif
#endif

#define SECRET_NUM -1234

#ifdef GPU

#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>

#ifdef CUDNN
#include <cudnn.h>
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct network;
typedef struct network network;

struct network_state;
typedef struct network_state network_state;

struct layer;
typedef struct layer layer;

struct image;
typedef struct image image;

struct detection;
typedef struct detection detection;

struct load_args;
typedef struct load_args load_args;

struct data;
typedef struct data data;

struct metadata;
typedef struct metadata metadata;

struct tree;
typedef struct tree tree;

extern int gpu_index;

// option_list.h
typedef struct metadata {
    int classes;
    char **names;
} metadata;


// tree.h
typedef struct tree {
    int *leaf;
    int n;
    int *parent;
    int *child;
    int *group;
    char **name;

    int groups;
    int *group_size;
    int *group_offset;
} tree;


// activations.h
//激活函数
typedef enum {
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN, SELU, SWISH, MISH, NORM_CHAN, NORM_CHAN_SOFTMAX, NORM_CHAN_SOFTMAX_MAXVAL
}ACTIVATION;

// parser.h
typedef enum {
    IOU, GIOU, MSE, DIOU, CIOU
} IOU_LOSS;

// parser.h
typedef enum {
    DEFAULT_NMS, GREEDY_NMS, DIOU_NMS, CORNERS_NMS
} NMS_KIND;

// parser.h
typedef enum {
    YOLO_CENTER = 1 << 0, YOLO_LEFT_TOP = 1 << 1, YOLO_RIGHT_BOTTOM = 1 << 2
} YOLO_POINT;

// parser.h
typedef enum {
    NO_WEIGHTS, PER_FEATURE, PER_CHANNEL
} WEIGHTS_TYPE_T;

// parser.h
typedef enum {
    NO_NORMALIZATION, RELU_NORMALIZATION, SOFTMAX_NORMALIZATION
} WEIGHTS_NORMALIZATION_T;

// image.h
typedef enum{
    PNG, BMP, TGA, JPG
} IMTYPE;

// activations.h
typedef enum{
    MULT, ADD, SUB, DIV
} BINARY_ACTIVATION;

// layer.h
typedef enum {
    CONVOLUTIONAL,
    DECONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    LOCAL_AVGPOOL,
    SOFTMAX,
    DETECTION,
    DROPOUT,
    CROP,
    ROUTE,
    COST,
    NORMALIZATION,
    AVGPOOL,
    LOCAL,
    SHORTCUT,
    SCALE_CHANNELS,
    SAM,
    ACTIVE,
    RNN,
    GRU,
    LSTM,
    CONV_LSTM,
    CRNN,
    BATCHNORM,
    NETWORK,
    XNOR,
    REGION,
    YOLO,
    GAUSSIAN_YOLO,
    ISEG,
    REORG,
    REORG_OLD,
    UPSAMPLE,
    LOGXENT,
    L2NORM,
    EMPTY,
    BLANK
} LAYER_TYPE;

// layer.h
typedef enum{
    SSE, MASKED, L1, SEG, SMOOTH,WGAN
} COST_TYPE;

// layer.h
typedef struct update_args {
    int batch;
    float learning_rate;
    float momentum;
    float decay;
    int adam;
    float B1;
    float B2;
    float eps;
    int t;
} update_args;

//定义layer
struct layer {
    LAYER_TYPE type; // 网络层的类型，枚举类型，取值比如DROPOUT,CONVOLUTIONAL,MAXPOOL分别表示dropout层，卷积层，最大池化层，可参见LAYER_TYPE枚举类型的定义
    ACTIVATION activation; //激活函数类型，枚举类型
    COST_TYPE cost_type; //损失函数类型，枚举类型
    void(*forward)   (struct layer, struct network_state);
    void(*backward)  (struct layer, struct network_state);
    void(*update)    (struct layer, int, float, float, float);
    void(*forward_gpu)   (struct layer, struct network_state);
    void(*backward_gpu)  (struct layer, struct network_state);
    void(*update_gpu)    (struct layer, int, float, float, float);
    layer *share_layer;
    int train;
    int avgpool;
    int batch_normalize; // 是否进行BN，如果进行BN，则值为1
    int shortcut;
    int batch; // 一个batch中含有的图片张数，等于net.batch，详细可以参考network.h中的注释，一般在构建具体网络层时赋值（比如make_maxpool_layer()中）
    int forced;
    int flipped;
    int inputs; // 一张输入图片所含的元素个数（一般在各网络层构建函数中赋值，比如make_connected_layer()），第一层的值等于l.h*l.w*l.c，
                // 之后的每一层都是由上一层的输出自动推算得到的（参见parse_network_cfg()，在构建每一层后，会更新params.inputs为上一层的l.outputs）
    int outputs; // 该层对应一张输入图片的输出元素个数（一般在各网络层构建函数中赋值，比如make_connected_layer()）
                 // 对于一些网络，可由输入图片的尺寸及相关参数计算出，比如卷积层，可以通过输入尺寸以及步长、核大小计算出；
                 // 对于另一些尺寸，则需要通过网络配置文件指定，如未指定，取默认值1，比如全连接层（见parse_connected()函数）
    int nweights;
    int nbiases;
    int extra;
    int truths;  // < 根据region_layer.c判断，这个变量表示一张图片含有的真实值的个数，对于检测模型来说，一个真实的标签含有5个值，
                // 包括类型对应的编号以及定位矩形框用到的w,h,x,y四个参数，且在darknet中固定每张图片最大处理30个矩形框，（可查看max_boxes参数），
                // 因此，在region_layer.c的make_region_layer()函数中赋值为30*5.
    int h, w, c;  // 该层输入的图片的宽，高，通道数（一般在各网络层构建函数中赋值，比如make_connected_layer()）
    int out_h, out_w, out_c;// 该层输出图片的高、宽、通道数（一般在各网络层构建函数中赋值，比如make_connected_layer()）
    int n; // 对于卷积层，该参数表示卷积核个数，等于out_c，其值由网络配置文件指定；对于region_layer层，该参数等于配置文件中的num值 
           // (该参数通过make_region_layer()函数赋值，在parser.c中调用的make_region_layer()函数)，
           // 可以在darknet/cfg文件夹下执行命令：grep num *.cfg便可以搜索出所有设置了num参数的网络，这里面包括yolo.cfg等，其值有
           // 设定为3,5,2的，该参数就是Yolo论文中的B，也就是一个cell中预测多少个box。
    int max_boxes; // 每张图片最多含有的标签矩形框数（参看：data.c中的load_data_detection()，其输入参数boxes就是指这个参数）， 
                   // 什么意思呢？就是每张图片中最多打了max_boxes个标签物体，模型预测过程中，可能会预测出很多的物体，但实际上，
                   // 图片中打上标签的真正存在的物体最多就max_boxes个，预测多出来的肯定存在false positive，需要滤出与筛选，
                   // 可参看region_layer.c中forward_region_layer()函数的第二个for循环中的注释
    int groups; // 应该是控制组卷积的组数，类似于caffe的group参数
    int group_id; 
    int size; // 核尺寸（比如卷积核，池化核等）
    int side;
    int stride; // 滑动步长，如卷积核的滑动步长
    int stride_x;
    int stride_y;
    int dilation; //空洞卷积参数
    int antialiasing; 
    int maxpool_depth;
    int out_channels; // 输出通道数
    int reverse;
    int flatten;
    int spatial;
    int pad; // 该层对输入数据四周的补0长度（现在发现在卷积层，最大池化层中有用到该参数），一般在构建具体网络层时赋值（比如make_maxpool_layer()中）
    int sqrt;
    int flip;
    int index;
    int scale_wh;
    int binary;
    int xnor;
    int peephole;
    int use_bin_output;
    int keep_delta_gpu;
    int optimized_memory;
    int steps;
    int state_constrain;
    int hidden;
    int truth;
    float smooth;
    float dot;
    int deform;
    int sway;
    int rotate;
    int stretch;
    int stretch_sway;
    float angle;
    float jitter;
    float saturation;
    float exposure;
    float shift;
    float ratio;
    float learning_rate_scale;
    float clip;
    int focal_loss;
    float *classes_multipliers;
    float label_smooth_eps;
    int noloss;
    int softmax;
    int classes; // 物体类别种数，一个训练好的网络，只能检测指定所有物体类别中的物体，比如yolo9000.cfg，设置该值为9418，
                 // 也就是该网络训练好了之后可以检测9418种物体。该参数由网络配置文件指定。目前在作者给的例子中,
                 // 有设置该值的配置文件大都是检测模型，纯识别的网络模型没有设置该值，我想是因为检测模型输出的一般会为各个类别的概率，
                 // 所以需要知道这个种类数目，而识别的话，不需要知道某个物体属于这些所有类的具体概率，因此可以不知道。
    int coords; // 这个参数一般用在检测模型中，且不是所有层都有这个参数，一般在检测模型最后一层有，比如region_layer层，该参数的含义
                // 是定位一个物体所需的参数个数，一般为4个，包括物体所在矩形框中心坐标x,y两个参数以及矩形框长宽w,h两个参数，
                // 可以在darknet/cfg文件夹下，执行grep coords *.cfg，会搜索出所有使用该参数的模型，并可看到该值都设置为4
    int background;
    int rescore;
    int objectness;
    int does_cost;
    int joint;
    int noadjust;
    int reorg;
    int log;
    int tanh;
    int *mask;
    int total;
    float bflops;

    int adam;
    float B1;
    float B2;
    float eps;

    int t;

    float alpha;
    float beta;
    float kappa;

    float coord_scale;
    float object_scale;
    float noobject_scale;
    float mask_scale;
    float class_scale;
    int bias_match;
    float random;
    float ignore_thresh;
    float truth_thresh;
    float iou_thresh;
    float thresh;
    float focus;
    int classfix;
    int absolute;
    int assisted_excitation;

    int onlyforward; // 标志参数，当值为1那么当前层只执行前向传播
    int stopbackward;  // 标志参数，用来强制停止反向传播过程（值为1则停止反向传播），参看network.c中的backward_network()函数 
    int dontload;
    int dontsave;
    int dontloadscales;
    int numload;

    float temperature; // 温度参数，softmax层特有参数，在parse_softmax()函数中赋值，由网络配置文件指定，如果未指定，则使用默认值1（见parse_softmax()）
    float probability; // dropout概率，即舍弃概率，相应的1-probability为保留概率（具体的使用可参见forward_dropout_layer()），在make_dropout_layer()中赋值，
						// 其值由网络配置文件指定，如果网络配置文件未指定，则取默认值0.5（见parse_dropout()）
    float dropblock_size_rel;  
    int dropblock_size_abs;
    int dropblock;
    float scale; // 在dropout层中，该变量是一个比例因子，取值为保留概率的倒数（darknet实现用的是inverted dropout），用于缩放输入元素的值
                       // （在网上随便搜索关于dropout的博客，都会提到inverted dropout），在make_dropout_layer()函数中赋值

    char  * cweights;
    int   * indexes; // 维度为l.out_h * l.out_w * l.out_c * l.batch，可知包含整个batch输入图片的输出，一般在构建具体网络层时动态分配内存（比如make_maxpool_layer()中）。
                     // 目前仅发现其用在在最大池化层中。该变量存储的是索引值，并与当前层所有输出元素一一对应，表示当前层每个输出元素的值是上一层输出中的哪一个元素值（存储的索引值是
                     // 在上一层所有输出元素（包含整个batch）中的索引），因为对于最大池化层，每一个输出元素的值实际是上一层输出（也即当前层输入）某个池化区域中的最大元素值，indexes就是记录
                     // 这些局部最大元素值在上一层所有输出元素中的总索引。记录这些值有什么用吗？当然有，用于反向传播过程计算上一层敏感度值，详见backward_maxpool_layer()以及forward_maxpool_layer()函数。
    int   * input_layers; //这个层有哪些输入层
    int   * input_sizes; // 输入层的尺寸
    float **layers_output; //产生的一系列输出层
    float **layers_delta; 
    WEIGHTS_TYPE_T weights_type;
    WEIGHTS_NORMALIZATION_T weights_normalizion;
	/* 
     * 这个参数用的不多，仅在region_layer.c中使用，该参数的作用是用于不同数据集间类别编号的转换，更为具体的，
     * 是coco数据集中80类物体编号与联合数据集中9000+物体类别编号之间的转换，可以对比查看data/coco.names与
     * data/9k.names以及data/coco9k.map三个文件（旧版的darknet可能没有，新版的darknet才有coco9k.map这个文件），
     * 可以发现，coco.names中每一个物体类别都可以在9k.names中找到,且coco.names中每个物体类别名称在9k.names
     * 中所在的行数就是coco9k.map中的编号值（减了1,因为在程序数组中编号从0开始），也就是这个map将coco数据集中
     * 的类别编号映射到联和数据集9k中的类别编号（这个9k数据集是一个联和多个数据集的大数集，其名称分类被层级划分，
     * ）（注意两个文件中物体的类别名称大部分都相同，有小部分存在小差异，虽然有差异，但只是两个数据集中使用的名称有所差异而已，
     * 对应的物体是一样的，比如在coco.names中摩托车的名称为motorbike，在联合数据集9k.names，其名称为motorcycle）.                   
    */
    int   * map; 
    int   * counts;
    float ** sums;
	// 这个参数目前只发现用在dropout层，用于存储一些列的随机数，这些随机数与dropout层的输入元素一一对应，维度为l.batch*l.inputs（包含整个batch的），在make_dropout_layer()函数中用calloc动态分配内存，
    // 并在前向传播函数forward_dropout_layer()函数中逐元素赋值。里面存储的随机数满足0~1均匀分布，干什么用呢？用于决定该输入元素的去留，
    // 我们知道dropout层就完成一个事：按照一定概率舍弃输入神经元（所谓舍弃就是置该输入的值为0），rand中存储的值就是如果小于l.probability，则舍弃该输入神经元（详见：forward_dropout_layer()）。
    // 为什么要保留这些随机数呢？和最大池化层中的l.indexes类似，在反向传播函数backward_dropout_layer()中用来指示计算上一层的敏感度值，因为dropout舍弃了一些输入，
    // 这些输入（dropout层的输入，上一层的输出）对应的敏感度值可以置为0，而那些没有舍弃的输入，才有必要由当前dropout层反向传播过去。
    float * rand; 
    float * cost;
    float * state;
    float * prev_state;
    float * forgot_state;
    float * forgot_delta;
    float * state_delta;
    float * combine_cpu;
    float * combine_delta_cpu;

    float *concat;
    float *concat_delta;

    float *binary_weights;

    float *biases;  // 当前层所有偏置，对于卷积层，维度l.n，每个卷积核有一个偏置；对于全连接层，维度等于单张输入图片对应的元素个数即outputs，一般在各网络构建函数中动态分配内存（比如make_connected_layer()
    float *bias_updates; // 当前层所有偏置更新值，对于卷积层，维度l.n，每个卷积核有一个偏置；对于全连接层，维度为outputs。所谓权重系数更新值，就是梯度下降中与步长相乘的那项，也即误差对偏置的导数，
                          // 一般在各网络构建函数中动态分配内存（比如make_connected_layer())

    float *scales;
    float *scale_updates;

    float *weights; //当前层所有权重系数（连接当前层和上一层的系数，但记在当前层上），对于卷积层，维度为l.n*l.c*l.size*l.size，即卷积核个数乘以卷积核尺寸再乘以输入通道数（各个通道上的权重系数独立不一样）；
                     // 对于全连接层，维度为单张图片输入与输出元素个数之积inputs*outputs，一般在各网络构建函数中动态分配内存（比如make_connected_layer()）
    float *weight_updates;// 当前层所有权重系数更新值，对于卷积层维度为l.n*l.c*l.size*l.size；对于全连接层，维度为单张图片输入与输出元素个数之积inputs*outputs，
                            // 所谓权重系数更新值，就是梯度下降中与步长相乘的那项，也即误差对权重的导数，一般在各网络构建函数中动态分配内存（比如make_connected_layer()

    float scale_x_y;
    float max_delta;
    float uc_normalizer;
    float iou_normalizer;
    float cls_normalizer;
    IOU_LOSS iou_loss;
    NMS_KIND nms_kind;
    float beta_nms;
    YOLO_POINT yolo_point;

    char *align_bit_weights_gpu;
    float *mean_arr_gpu;
    float *align_workspace_gpu;
    float *transposed_align_workspace_gpu;
    int align_workspace_size;

    char *align_bit_weights;
    float *mean_arr;
    int align_bit_weights_size;
    int lda_align;
    int new_lda;
    int bit_align;

    float *col_image;
    float * delta; // 存储每一层的敏感度图：包含所有输出元素的敏感度值（整个batch所有图片）。所谓敏感度，即误差函数关于当前层每个加权输入的导数值，
                   // 关于敏感度图这个名称，其实就是梯度，可以参考https://www.zybuluo.com/hanbingtao/note/485480。
                   // 元素个数为l.batch * l.outputs（其中l.outputs = l.out_h * l.out_w * l.out_c），
                   // 对于卷积神经网络，在make_convolutional_layer()动态分配内存，按行存储，可视为l.batch行，l.outputs列，
                   // 即batch中每一张图片，对应l.delta中的一行，而这一行，又可以视作有l.out_c行，l.out_h*l.out_c列，
                   // 其中每小行对应一张输入图片的一张输出特征图的敏感度。一般在构建具体网络层时动态分配内存（比如make_maxpool_layer()中）。
    float * output; // 存储该层所有的输出，维度为l.out_h * l.out_w * l.out_c * l.batch，可知包含整个batch输入图片的输出，一般在构建具体网络层时动态分配内存（比如make_maxpool_layer()中）。
                    // 按行存储：每张图片按行铺排成一大行，图片间再并成一行。
    float * activation_input;
    int delta_pinned;
    int output_pinned;
    float * loss;
    float * squared;
    float * norms;

    float * spatial_mean;
    float * mean;
    float * variance;

    float * mean_delta;
    float * variance_delta;

    float * rolling_mean;
    float * rolling_variance;

    float * x;
    float * x_norm;

    float * m;
    float * v;

    float * bias_m;
    float * bias_v;
    float * scale_m;
    float * scale_v;


    float *z_cpu;
    float *r_cpu;
    float *h_cpu;
    float *stored_h_cpu;
    float * prev_state_cpu;

    float *temp_cpu;
    float *temp2_cpu;
    float *temp3_cpu;

    float *dh_cpu;
    float *hh_cpu;
    float *prev_cell_cpu;
    float *cell_cpu;
    float *f_cpu;
    float *i_cpu;
    float *g_cpu;
    float *o_cpu;
    float *c_cpu;
    float *stored_c_cpu;
    float *dc_cpu;

    float *binary_input;
    uint32_t *bin_re_packed_input;
    char *t_bit_input;

    struct layer *input_layer;
    struct layer *self_layer;
    struct layer *output_layer;

    struct layer *reset_layer;
    struct layer *update_layer;
    struct layer *state_layer;

    struct layer *input_gate_layer;
    struct layer *state_gate_layer;
    struct layer *input_save_layer;
    struct layer *state_save_layer;
    struct layer *input_state_layer;
    struct layer *state_state_layer;

    struct layer *input_z_layer;
    struct layer *state_z_layer;

    struct layer *input_r_layer;
    struct layer *state_r_layer;

    struct layer *input_h_layer;
    struct layer *state_h_layer;

    struct layer *wz;
    struct layer *uz;
    struct layer *wr;
    struct layer *ur;
    struct layer *wh;
    struct layer *uh;
    struct layer *uo;
    struct layer *wo;
    struct layer *vo;
    struct layer *uf;
    struct layer *wf;
    struct layer *vf;
    struct layer *ui;
    struct layer *wi;
    struct layer *vi;
    struct layer *ug;
    struct layer *wg;
 
    tree *softmax_tree; // softmax层用到的一个参数，不过这个参数似乎并不常见，很多用到softmax层的网络并没用使用这个参数，目前仅发现darknet9000.cfg中使用了该参数，如果未用到该参数，其值为NULL，如果用到了则会在parse_softmax()中赋值，
                       // 目前个人的初步猜测是利用该参数来组织标签数据，以方便访问

    size_t workspace_size; // net.workspace的元素个数，为所有层中最大的l.out_h*l.out_w*l.size*l.size*l.c，（在make_convolutional_layer()计算得到workspace_size的大小，在parse_network_cfg()中动态分配内存，此值对应未使用gpu时的情况
 
#ifdef GPU
    int *indexes_gpu;

    float *z_gpu;
    float *r_gpu;
    float *h_gpu;
    float *stored_h_gpu;

    float *temp_gpu;
    float *temp2_gpu;
    float *temp3_gpu;

    float *dh_gpu;
    float *hh_gpu;
    float *prev_cell_gpu;
    float *prev_state_gpu;
    float *last_prev_state_gpu;
    float *last_prev_cell_gpu;
    float *cell_gpu;
    float *f_gpu;
    float *i_gpu;
    float *g_gpu;
    float *o_gpu;
    float *c_gpu;
    float *stored_c_gpu;
    float *dc_gpu;

    // adam
    float *m_gpu;
    float *v_gpu;
    float *bias_m_gpu;
    float *scale_m_gpu;
    float *bias_v_gpu;
    float *scale_v_gpu;

    float * combine_gpu;
    float * combine_delta_gpu;

    float * forgot_state_gpu;
    float * forgot_delta_gpu;
    float * state_gpu;
    float * state_delta_gpu;
    float * gate_gpu;
    float * gate_delta_gpu;
    float * save_gpu;
    float * save_delta_gpu;
    float * concat_gpu;
    float * concat_delta_gpu;

    float *binary_input_gpu;
    float *binary_weights_gpu;
    float *bin_conv_shortcut_in_gpu;
    float *bin_conv_shortcut_out_gpu;

    float * mean_gpu;
    float * variance_gpu;

    float * rolling_mean_gpu;
    float * rolling_variance_gpu;

    float * variance_delta_gpu;
    float * mean_delta_gpu;

    float * col_image_gpu;

    float * x_gpu;
    float * x_norm_gpu;
    float * weights_gpu;
    float * weight_updates_gpu;
    float * weight_deform_gpu;
    float * weight_change_gpu;

    float * weights_gpu16;
    float * weight_updates_gpu16;

    float * biases_gpu;
    float * bias_updates_gpu;
    float * bias_change_gpu;

    float * scales_gpu;
    float * scale_updates_gpu;
    float * scale_change_gpu;

    float * input_antialiasing_gpu;
    float * output_gpu;
    float * activation_input_gpu;
    float * loss_gpu;
    float * delta_gpu;
    float * rand_gpu;
    float * squared_gpu;
    float * norms_gpu;

    float *gt_gpu;
    float *a_avg_gpu;

    int *input_sizes_gpu;
    float **layers_output_gpu;
    float **layers_delta_gpu;
#ifdef CUDNN
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
    cudnnTensorDescriptor_t srcTensorDesc16, dstTensorDesc16;
    cudnnTensorDescriptor_t dsrcTensorDesc, ddstTensorDesc;
    cudnnTensorDescriptor_t dsrcTensorDesc16, ddstTensorDesc16;
    cudnnTensorDescriptor_t normTensorDesc, normDstTensorDesc, normDstTensorDescF16;
    cudnnFilterDescriptor_t weightDesc, weightDesc16;
    cudnnFilterDescriptor_t dweightDesc, dweightDesc16;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t fw_algo, fw_algo16;
    cudnnConvolutionBwdDataAlgo_t bd_algo, bd_algo16;
    cudnnConvolutionBwdFilterAlgo_t bf_algo, bf_algo16;
    cudnnPoolingDescriptor_t poolingDesc;
#endif  // CUDNN
#endif  // GPU
};


// network.h
typedef enum {
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM, SGDR
} learning_rate_policy;

// network.h
// 定义network结构体
typedef struct network {
    int n; //网络的层数，调用make_network(int n)时赋值
    int batch; //一批训练中的图片参数，和subdivsions参数相关
    uint64_t *seen; //目前已经读入的图片张数(网络已经处理的图片张数) 
    int *t; 
    float epoch; //到目前为止训练了整个数据集的次数
    int subdivisions;
    layer *layers; //存储网络中的所有层  
    float *output;
    learning_rate_policy policy; // 学习率下降策略
    int benchmark_layers; 
    // 梯度下降法相关参数  
    float learning_rate; //学习率
    float learning_rate_min; //学习率最小值
    float learning_rate_max;  //学习率最大值
    int batches_per_cycle; //
    int batches_cycle_mult;
    float momentum;
    float decay;
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    int num_boxes;
    int train_images_num;
    float *seq_scales;
    float *scales;
    int   *steps;
    int num_steps;
    int burn_in;
    int cudnn_half;
    // ADAM优化方法相关策略
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
    int flip; // horizontal flip 50% probability augmentaiont for classifier training (default = 1)
    int blur;
    int mixup;
    float label_smooth_eps;
    int resize_step;
    int letter_box;
    float angle;
    float aspect;
    float exposure;
    float saturation;
    float hue;
    int random;
    int track;
    int augment_speed;
    int sequential_subdivisions;
    int init_sequential_subdivisions;
    int current_subdivision;
    int try_fix_nan;
    //darknet 为每个 GPU 维护一个相同的 network, 每个 network 以 gpu_index 区分
    int gpu_index;
    tree *hierarchy;

	//中间变量，用来暂存某层网络的输入（包含一个 batch 的输入，比如某层网络完成前向，
    //将其输出赋给该变量，作为下一层的输入，可以参看 network.c 中的forward_network() 
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
	// 标志参数，当前网络的活跃层 
    int index;
	//每一层的损失，只有[yolo]层有值
    float *cost;
    float clip;

#ifdef GPU
    //float *input_gpu;
    //float *truth_gpu;
    float *delta_gpu;
    float *output_gpu;

    float *input_state_gpu;
    float *input_pinned_cpu;
    int input_pinned_cpu_flag;

    float **input_gpu;
    float **truth_gpu;
    float **input16_gpu;
    float **output16_gpu;
    size_t *max_input16_size;
    size_t *max_output16_size;
    int wait_stream;

    float *global_delta_gpu;
    float *state_delta_gpu;
    size_t max_delta_gpu_size;
#endif
    int optimized_memory;
    size_t workspace_size_limit;
} network;

// network.h
typedef struct network_state {
    float *truth;
    float *input;
    float *delta;
    float *workspace;
    int train;
    int index;
    network net;
} network_state;

//typedef struct {
//    int w;
//    int h;
//    float scale;
//    float rad;
//    float dx;
//    float dy;
//    float aspect;
//} augment_args;

// image.h
typedef struct image {
    int w;
    int h;
    int c;
    float *data;
} image;

//typedef struct {
//    int w;
//    int h;
//    int c;
//    float *data;
//} image;

// box.h
typedef struct box {
    float x, y, w, h;
} box;

// box.h
typedef struct boxabs {
    float left, right, top, bot;
} boxabs;

// box.h
typedef struct dxrep {
    float dt, db, dl, dr;
} dxrep;

// box.h
typedef struct ious {
    float iou, giou, diou, ciou;
    dxrep dx_iou;
    dxrep dx_giou;
} ious;


// box.h
typedef struct detection{
    box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
    float *uc; // Gaussian_YOLOv3 - tx,ty,tw,th uncertainty
    int points; // bit-0 - center, bit-1 - top-left-corner, bit-2 - bottom-right-corner
} detection;

// matrix.h
typedef struct matrix {
    int rows, cols;
    float **vals;
} matrix;

// data.h
typedef struct data {
    int w, h;
    matrix X;
    matrix y;
    int shallow;
    int *num_boxes;
    box **boxes;
} data;

// data.h
typedef enum {
    CLASSIFICATION_DATA, DETECTION_DATA, CAPTCHA_DATA, REGION_DATA, IMAGE_DATA, COMPARE_DATA, WRITING_DATA, SWAG_DATA, TAG_DATA, OLD_CLASSIFICATION_DATA, STUDY_DATA, DET_DATA, SUPER_DATA, LETTERBOX_DATA, REGRESSION_DATA, SEGMENTATION_DATA, INSTANCE_DATA, ISEG_DATA
} data_type;

// data.h
typedef struct load_args {
    int threads;
    char **paths;
    char *path;
    int n;
    int m;
    char **labels;
    int h;
    int w;
    int c; // color depth
    int out_w;
    int out_h;
    int nh;
    int nw;
    int num_boxes;
    int min, max, size;
    int classes;
    int background;
    int scale;
    int center;
    int coords;
    int mini_batch;
    int track;
    int augment_speed;
    int letter_box;
    int show_imgs;
    int dontuse_opencv;
    float jitter;
    int flip;
    int blur;
    int mixup;
    float label_smooth_eps;
    float angle;
    float aspect;
    float saturation;
    float exposure;
    float hue;
    data *d;
    image *im;
    image *resized;
    data_type type;
    tree *hierarchy;
} load_args;

// data.h
typedef struct box_label {
    int id;
    float x, y, w, h;
    float left, right, top, bottom;
} box_label;

// list.h
//typedef struct node {
//    void *val;
//    struct node *next;
//    struct node *prev;
//} node;

// list.h
//typedef struct list {
//    int size;
//    node *front;
//    node *back;
//} list;

// -----------------------------------------------------


// parser.c
LIB_API network *load_network(char *cfg, char *weights, int clear);
LIB_API network *load_network_custom(char *cfg, char *weights, int clear, int batch);
LIB_API network *load_network(char *cfg, char *weights, int clear);
LIB_API void free_network(network net);

// network.c
LIB_API load_args get_base_args(network *net);

// box.h
LIB_API void do_nms_sort(detection *dets, int total, int classes, float thresh);
LIB_API void do_nms_obj(detection *dets, int total, int classes, float thresh);
LIB_API void diounms_sort(detection *dets, int total, int classes, float thresh, NMS_KIND nms_kind, float beta1);

// network.h
LIB_API float *network_predict(network net, float *input);
LIB_API float *network_predict_ptr(network *net, float *input);
LIB_API detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num, int letter);
LIB_API void free_detections(detection *dets, int n);
LIB_API void fuse_conv_batchnorm(network net);
LIB_API void calculate_binary_weights(network net);
LIB_API char *detection_to_json(detection *dets, int nboxes, int classes, char **names, long long int frame_id, char *filename);

LIB_API layer* get_network_layer(network* net, int i);
//LIB_API detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num, int letter);
LIB_API detection *make_network_boxes(network *net, float thresh, int *num);
LIB_API void reset_rnn(network *net);
LIB_API float *network_predict_image(network *net, image im);
LIB_API float *network_predict_image_letterbox(network *net, image im);
LIB_API float validate_detector_map(char *datacfg, char *cfgfile, char *weightfile, float thresh_calc_avg_iou, const float iou_thresh, const int map_points, int letter_box, network *existing_net);
LIB_API void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, int dont_show, int calc_map, int mjpeg_port, int show_imgs, int benchmark_layers);
LIB_API void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers);
LIB_API int network_width(network *net);
LIB_API int network_height(network *net);
LIB_API void optimize_picture(network *net, image orig, int max_layer, float scale, float rate, float thresh, int norm);

// image.h
LIB_API image resize_image(image im, int w, int h);
LIB_API void copy_image_from_bytes(image im, char *pdata);
LIB_API image letterbox_image(image im, int w, int h);
LIB_API void rgbgr_image(image im);
LIB_API image make_image(int w, int h, int c);
LIB_API image load_image_color(char *filename, int w, int h);
LIB_API void free_image(image m);

// layer.h
LIB_API void free_layer_custom(layer l, int keep_cudnn_desc);
LIB_API void free_layer(layer l);

// data.c
LIB_API void free_data(data d);
LIB_API pthread_t load_data(load_args args);
LIB_API pthread_t load_data_in_thread(load_args args);
LIB_API void *load_thread(void *ptr);

// dark_cuda.h
LIB_API void cuda_pull_array(float *x_gpu, float *x, size_t n);
LIB_API void cuda_pull_array_async(float *x_gpu, float *x, size_t n);
LIB_API void cuda_set_device(int n);
LIB_API void *cuda_get_context();

// utils.h
LIB_API void free_ptrs(void **ptrs, int n);
LIB_API void top_k(float *a, int n, int k, int *index);

// tree.h
LIB_API tree *read_tree(char *filename);

// option_list.h
LIB_API metadata get_metadata(char *file);


// http_stream.h
LIB_API void delete_json_sender();
LIB_API void send_json_custom(char const* send_buf, int port, int timeout);
LIB_API double get_time_point();
void start_timer();
void stop_timer();
double get_time();
void stop_timer_and_show();
void stop_timer_and_show_name(char *name);
void show_total_time();

// gemm.h
LIB_API void init_cpu();

#ifdef __cplusplus
}
#endif  // __cplusplus
#endif  // DARKNET_API
