#include "convolutional_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include "box.h"
#include <stdio.h>
#include <time.h>

#ifdef AI2
#include "xnor_layer.h"
#endif

#ifdef __cplusplus
#define PUT_IN_REGISTER
#else
#define PUT_IN_REGISTER register
#endif

#ifndef AI2
#define AI2 0
void forward_xnor_layer(layer l, network_state state);
#endif

void swap_binary(convolutional_layer *l)
{
    float *swap = l->weights;
    l->weights = l->binary_weights;
    l->binary_weights = swap;

    #ifdef GPU
    swap = l->weights_gpu;
    l->weights_gpu = l->binary_weights_gpu;
    l->binary_weights_gpu = swap;
    #endif
}

void binarize_weights(float *weights, int n, int size, float *binary)
{
    int i, f;
    for(f = 0; f < n; ++f){
        float mean = 0;
        for(i = 0; i < size; ++i){
            mean += fabs(weights[f*size + i]);
        }
        mean = mean / size;
        for(i = 0; i < size; ++i){
            binary[f*size + i] = (weights[f*size + i] > 0) ? mean: -mean;
        }
    }
}

void binarize_cpu(float *input, int n, float *binary)
{
    int i;
    for(i = 0; i < n; ++i){
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}

void binarize_input(float *input, int n, int size, float *binary)
{
    int i, s;
    for(s = 0; s < size; ++s){
        float mean = 0;
        for(i = 0; i < n; ++i){
            mean += fabs(input[i*size + s]);
        }
        mean = mean / n;
        for(i = 0; i < n; ++i){
            binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
        }
    }
}

/*
** 根据输入图像的高度，pad，卷积核尺寸以及步长计算输出的特征图的高度
*/
int convolutional_out_height(convolutional_layer l)
{
	// pad是每边补0的个数
    // 当stride=1, pad=size/2(整数除法，会向下取整)时, 输出高度等于输入高度(same策略)
    // 当stride=1,pad=0时，为valid策略
    // 当stride不等于1时，输出高度恒小于输入高度（尺寸一定会缩小）
    // 计算公式推导：设输出高度为x，总图像高度为h+2*pad个像素，输出高度为x，则共有x-1次卷积核移位，
    // 共占有(x-1)*stride+size个像素，可能还剩余res个像素，且res一定小于stride（否则还可以再移位一次），
    // 因此有(x-1)*stride+size+res=h+2*pad，->x=(h+2*pad-size)/stride+1-res/stride，因为res<stride，
    // 对于整数除法来说，值为0,于是得到最终的输出高度为x=(h+2*pad-size)/stride+1
    return (l.h + 2*l.pad - l.size) / l.stride_y + 1;
}


/*
** 和上个函数原理一样
*/
int convolutional_out_width(convolutional_layer l)
{
    return (l.w + 2*l.pad - l.size) / l.stride_x + 1;
}

image get_convolutional_image(convolutional_layer l)
{
    int h,w,c;
    h = convolutional_out_height(l);
    w = convolutional_out_width(l);
    c = l.n;
    return float_to_image(w,h,c,l.output);
}

image get_convolutional_delta(convolutional_layer l)
{
    int h,w,c;
    h = convolutional_out_height(l);
    w = convolutional_out_width(l);
    c = l.n;
    return float_to_image(w,h,c,l.delta);
}

size_t get_workspace_size32(layer l){
#ifdef CUDNN
    if(gpu_index >= 0){
        size_t most = 0;
        size_t s = 0;
        CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.weightDesc,
                l.convDesc,
                l.dstTensorDesc,
                l.fw_algo,
                &s));
        if (s > most) most = s;
        CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dweightDesc,
                l.bf_algo,
                &s));
        if (s > most && l.train) most = s;
        CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
                l.weightDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dsrcTensorDesc,
                l.bd_algo,
                &s));
        if (s > most && l.train) most = s;
        return most;
    }
    #endif
    if (l.xnor) {
        size_t re_packed_input_size = l.c * l.w * l.h * sizeof(float);
        size_t workspace_size = (size_t)l.bit_align*l.size*l.size*l.c * sizeof(float);
        if (workspace_size < re_packed_input_size) workspace_size = re_packed_input_size;
        return workspace_size;
    }
    return (size_t)l.out_h*l.out_w*l.size*l.size*(l.c / l.groups)*sizeof(float);
}

size_t get_workspace_size16(layer l) {
#if defined(CUDNN) && defined(CUDNN_HALF)
    if (gpu_index >= 0) {
        size_t most = 0;
        size_t s = 0;
        CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
            l.srcTensorDesc16,
            l.weightDesc16,
            l.convDesc,
            l.dstTensorDesc16,
            l.fw_algo16,
            &s));
        if (s > most) most = s;
        CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
            l.srcTensorDesc16,
            l.ddstTensorDesc16,
            l.convDesc,
            l.dweightDesc16,
            l.bf_algo16,
            &s));
        if (s > most && l.train) most = s;
        CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
            l.weightDesc16,
            l.ddstTensorDesc16,
            l.convDesc,
            l.dsrcTensorDesc16,
            l.bd_algo16,
            &s));
        if (s > most && l.train) most = s;
        return most;
    }
#endif
    return 0;
    //if (l.xnor) return (size_t)l.bit_align*l.size*l.size*l.c * sizeof(float);
    //return (size_t)l.out_h*l.out_w*l.size*l.size*l.c * sizeof(float);
}

size_t get_convolutional_workspace_size(layer l) {
    size_t workspace_size = get_workspace_size32(l);
    size_t workspace_size16 = get_workspace_size16(l);
    if (workspace_size16 > workspace_size) workspace_size = workspace_size16;
    return workspace_size;
}
#ifdef GPU
#ifdef CUDNN
void create_convolutional_cudnn_tensors(layer *l)
{
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->normTensorDesc));

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->normDstTensorDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->srcTensorDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->dstTensorDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&l->weightDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->dsrcTensorDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->ddstTensorDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&l->dweightDesc));

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->normDstTensorDescF16));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->srcTensorDesc16));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->dstTensorDesc16));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&l->weightDesc16));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->dsrcTensorDesc16));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->ddstTensorDesc16));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&l->dweightDesc16));

    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&l->convDesc));
}

void cudnn_convolutional_setup(layer *l, int cudnn_preference, size_t workspace_size_specify)
{

// CUDNN_HALF
    // TRUE_HALF_CONFIG is only supported on architectures with true fp16 support (compute capability 5.3 and 6.0):
    //   Tegra X1, Jetson TX1, DRIVE CX, DRIVE PX, Quadro GP100, Tesla P100
    // PSEUDO_HALF_CONFIG is required for Tensor Cores - our case!

    cudnnDataType_t data_type = CUDNN_DATA_FLOAT;

#if(CUDNN_MAJOR >= 7)
    // Tensor Core uses CUDNN_TENSOR_OP_MATH instead of CUDNN_DEFAULT_MATH
    // For *_ALGO_WINOGRAD_NONFUSED can be used CUDNN_DATA_FLOAT
    // otherwise Input, Filter and Output descriptors (xDesc, yDesc, wDesc, dxDesc, dyDesc and dwDesc as applicable) have dataType = CUDNN_DATA_HALF
    // Three techniques for training using Mixed-precision: https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/
    // 1. Accumulation into FP32
    // 2. Loss Scaling - required only for: activation gradients. We do not use.
    // 3. FP32 Master Copy of Weights
    // More: http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#tensor_ops
    CHECK_CUDNN(cudnnSetConvolutionGroupCount(l->convDesc, l->groups));
    CHECK_CUDNN(cudnnSetConvolutionMathType(l->convDesc, CUDNN_TENSOR_OP_MATH));
#if((CUDNN_MAJOR*10 + CUDNN_MINOR) >= 72)   // cuDNN >= 7.2
    //CHECK_CUDNN(cudnnSetConvolutionMathType(l->convDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION)); // reduces the speed of regular and group convolution
#endif
#else   //if(CUDNN_MAJOR >= 7)
    if (l->groups > 1) {
        error("CUDNN < 7 doesn't support groups, please upgrade!");
    }
#endif

    // INT8_CONFIG, INT8_EXT_CONFIG, INT8x4_CONFIG and INT8x4_EXT_CONFIG are only supported
    //   on architectures with DP4A support (compute capability 6.1 and later).
    //cudnnDataType_t data_type = CUDNN_DATA_INT8;

    // backward delta
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->dsrcTensorDesc, CUDNN_TENSOR_NCHW, data_type, l->batch, l->c, l->h, l->w));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->ddstTensorDesc, CUDNN_TENSOR_NCHW, data_type, l->batch, l->out_c, l->out_h, l->out_w));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(l->dweightDesc, data_type, CUDNN_TENSOR_NCHW, l->n, l->c / l->groups, l->size, l->size));

    // forward
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, data_type, l->batch, l->c, l->h, l->w));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, data_type, l->batch, l->out_c, l->out_h, l->out_w));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(l->weightDesc, data_type, CUDNN_TENSOR_NCHW, l->n, l->c / l->groups, l->size, l->size));

//#ifdef CUDNN_HALF
    // backward delta
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->dsrcTensorDesc16, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, l->batch, l->c, l->h, l->w));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->ddstTensorDesc16, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, l->batch, l->out_c, l->out_h, l->out_w));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(l->dweightDesc16, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, l->n, l->c / l->groups, l->size, l->size));

    // forward
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->srcTensorDesc16, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, l->batch, l->c, l->h, l->w));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->dstTensorDesc16, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, l->batch, l->out_c, l->out_h, l->out_w));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(l->weightDesc16, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, l->n, l->c / l->groups, l->size, l->size));

    // batch norm
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->normDstTensorDescF16, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, l->batch, l->out_c, l->out_h, l->out_w));
//#endif

    // batch norm
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->normDstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w));

    //printf("\n l->dilation = %d, l->pad = %d, l->size = %d \n", l->dilation, l->pad, l->size);
#if(CUDNN_MAJOR >= 6)
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(l->convDesc, l->pad * l->dilation, l->pad* l->dilation, l->stride_y, l->stride_x, l->dilation, l->dilation, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));    // cudnn >= 6.0
#else
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(l->convDesc, l->pad * l->dilation, l->pad * l->dilation, l->stride_y, l->stride_x, l->dilation, l->dilation, CUDNN_CROSS_CORRELATION));    // cudnn 5.1
#endif
    int forward_algo = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
    int backward_algo = CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST;
    int backward_filter = CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST;
    if (cudnn_preference == cudnn_smallest)
    {
        forward_algo = CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;
        backward_algo = CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE;
        backward_filter = CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE;
        printf(" CUDNN-slow ");
    }
    if (cudnn_preference == cudnn_specify)
    {
        forward_algo = CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT;
        backward_algo = CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT;
        backward_filter = CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT;
        //printf(" CUDNN-specified %zu ", workspace_size_specify);
    }

    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->weightDesc,
            l->convDesc,
            l->dstTensorDesc,
            (cudnnConvolutionFwdPreference_t)forward_algo,
            workspace_size_specify,
            &l->fw_algo));
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle(),
            l->weightDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dsrcTensorDesc,
            (cudnnConvolutionBwdDataPreference_t)backward_algo,
            workspace_size_specify,
            &l->bd_algo));
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dweightDesc,
            (cudnnConvolutionBwdFilterPreference_t)backward_filter,
            workspace_size_specify,
            &l->bf_algo));

    //if (data_type == CUDNN_DATA_HALF)
    {
        // HALF-16 if(data_type == CUDNN_DATA_HALF)
        l->fw_algo16 = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
        l->bd_algo16 = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
        l->bf_algo16 = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

        // FLOAT-32 if(data_type == CUDNN_DATA_FLOAT)
        //l->fw_algo16 = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
        //l->bd_algo16 = CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED;
        //l->bf_algo16 = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED;
    }
}
#endif
#endif


void free_convolutional_batchnorm(convolutional_layer *l)
{
    if (!l->share_layer) {
        free(l->scales);            l->scales = NULL;
        free(l->scale_updates);     l->scale_updates = NULL;
        free(l->mean);              l->mean = NULL;
        free(l->variance);          l->variance = NULL;
        free(l->mean_delta);        l->mean_delta = NULL;
        free(l->variance_delta);    l->variance_delta = NULL;
        free(l->rolling_mean);      l->rolling_mean = NULL;
        free(l->rolling_variance);  l->rolling_variance = NULL;
        free(l->x);                 l->x = NULL;
        free(l->x_norm);            l->x_norm = NULL;

#ifdef GPU
        cuda_free(l->scales_gpu);           l->scales_gpu = NULL;
        cuda_free(l->scale_updates_gpu);    l->scale_updates_gpu = NULL;
        cuda_free(l->mean_gpu);             l->mean_gpu = NULL;
        cuda_free(l->variance_gpu);         l->variance_gpu = NULL;
        cuda_free(l->mean_delta_gpu);       l->mean_delta_gpu = NULL;
        cuda_free(l->variance_delta_gpu);   l->variance_delta_gpu = NULL;
        cuda_free(l->rolling_mean_gpu);     l->rolling_mean_gpu = NULL;
        cuda_free(l->rolling_variance_gpu); l->rolling_variance_gpu = NULL;
        cuda_free(l->x_gpu);                l->x_gpu = NULL;
        cuda_free(l->x_norm_gpu);           l->x_norm_gpu = NULL;
#endif
    }
}


/*
** batch 每个batch含有的图片数
** step 
** h 图像高度(行数)
** w 图像宽度(列数)
** c 输入图像通道数
** n 卷积核个数
** groups 分组数
** size 卷积核尺寸
** stride 步长
** dilation 空洞卷积空洞率
** padding 四周补0长度
** activation 激活函数类别
** batch_normalize 是否进行BN
** binary 是否对权重进行二值化
** xnor 是否对权重以及输入进行二值化
** adam 优化方式
** use_bin_output 
** index 分组卷积的时候分组索引
** antialiasing 抗锯齿标志，如果为真强行设置所有的步长为1
** share_layer 标志参数，表示这一个卷积层是否和其它卷积层贡献权重
** assisted_excitation
** deform 暂时不知道
** train 标志参数，是否在训练
*/
convolutional_layer make_convolutional_layer(int batch, int steps, int h, int w, int c, int n, int groups, int size, int stride_x, int stride_y, int dilation, int padding, ACTIVATION activation,
 int batch_normalize, int binary, int xnor, int adam, int use_bin_output, int index, int antialiasing, convolutional_layer *share_layer, int assisted_excitation, int deform, int train)
{
    int total_batch = batch*steps;
    int i;
    convolutional_layer l = { (LAYER_TYPE)0 };
    l.type = CONVOLUTIONAL;
    l.train = train;

    if (xnor) groups = 1;   //对于二值网络，不能使用分组卷积
    if (groups < 1) groups = 1;

    const int blur_stride_x = stride_x;
    const int blur_stride_y = stride_y;
    l.antialiasing = antialiasing;
    if (antialiasing) {
        stride_x = stride_y = l.stride = l.stride_x = l.stride_y = 1; // use stride=1 in host-layer
    }

    l.deform = deform;
    l.assisted_excitation = assisted_excitation;
    l.share_layer = share_layer;
    l.index = index;
    l.h = h;
    l.w = w;
    l.c = c;
    l.groups = groups;
    l.n = n;
    l.binary = binary;
    l.xnor = xnor;
    l.use_bin_output = use_bin_output;
    l.batch = batch;
    l.steps = steps;
    l.stride = stride_x;
    l.stride_x = stride_x;
    l.stride_y = stride_y;
    l.dilation = dilation;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;
    l.learning_rate_scale = 1;
	// 该卷积层总的权重元素个数（权重元素个数等于输入数据的通道数/分组数*卷积核个数*卷积核的二维尺寸，注意因为每一个卷积核是同时作用于输入数据
    // 的多个通道上的，因此实际上卷积核是三维的，包括两个维度的平面尺寸，以及输入数据通道数这个维度，每个通道上的卷积核参数都是独立的训练参数）
    l.nweights = (c / groups) * n * size * size;
	// 如果是共享卷积层，可以直接用共享的卷积层来赋值（猜测是有预训练权重的时候可以直接赋值）
    if (l.share_layer) {
        if (l.size != l.share_layer->size || l.nweights != l.share_layer->nweights || l.c != l.share_layer->c || l.n != l.share_layer->n) {
            printf("Layer size, nweights, channels or filters don't match for the share_layer");
            getchar();
        }

        l.weights = l.share_layer->weights;
        l.weight_updates = l.share_layer->weight_updates;

        l.biases = l.share_layer->biases;
        l.bias_updates = l.share_layer->bias_updates;
    }
    else {
			// 该卷积层总的权重元素(卷积核元素)个数=输入图像通道数 / 分组数*卷积核个数*卷积核尺寸
        l.weights = (float*)xcalloc(l.nweights, sizeof(float));
		// bias就是Wx+b中的b（上面的weights就是W），有多少个卷积核，就有多少个b（与W的个数一一对应，每个W的元素个数为c*size*size）
        l.biases = (float*)xcalloc(n, sizeof(float));
		// 训练期间，需要执行反向传播
        if (train) {
			// 敏感图和特征图的尺寸应该是一样的
            l.weight_updates = (float*)xcalloc(l.nweights, sizeof(float));
			// bias的敏感图，维度和bias一致
            l.bias_updates = (float*)xcalloc(n, sizeof(float));
        }
    }

    // float scale = 1./sqrt(size*size*c);
	// 初始化权重：缩放因子*标准正态分布随机数，缩放因子等于sqrt(2./(size*size*c))，随机初始化
    // 此处初始化权重为正态分布，而在全连接层make_connected_layer()中初始化权重是均匀分布的。
    // TODO：个人感觉，这里应该加一个if条件语句：if(weightfile)，因为如果导入了预训练权重文件，就没有必要这样初始化了（事实上在detector.c的train_detector()函数中，
    // 紧接着parse_network_cfg()函数之后，就添加了if(weightfile)语句判断是否导入权重系数文件，如果导入了权重系数文件，也许这里初始化的值也会覆盖掉，
    // 总之这里的权重初始化的处理方式还是值得思考的，也许更好的方式是应该设置专门的函数进行权重的初始化，同时偏置也是，不过这里似乎没有考虑偏置的初始化，在make_connected_layer()中倒是有。。。）
    float scale = sqrt(2./(size*size*c/groups));
    for(i = 0; i < l.nweights; ++i) l.weights[i] = scale*rand_uniform(-1, 1);   // rand_normal();
	// 根据该层输入图像的尺寸、卷积核尺寸以及跨度计算输出特征图的宽度和高度
    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
	// 输出图像高度
    l.out_h = out_h;
	// 输出图像宽度	
    l.out_w = out_w;
	// 输出图像通道数(等于卷积核个数,有多少个卷积核，最终就得到多少张特征图，每张特征图是一个通道)
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c; // 对应每张输入图片的所有输出特征图的总元素个数（每张输入图片会得到n也即l.out_c张特征图）
    l.inputs = l.w * l.h * l.c; // mini-batch中每张输入图片的像素元素个数
    l.activation = activation;

    l.output = (float*)xcalloc(total_batch*l.outputs, sizeof(float)); // l.output为该层所有的输出（包括mini-batch所有输入图片的输出）
#ifndef GPU
    if (train) l.delta = (float*)xcalloc(total_batch*l.outputs, sizeof(float));  // l.delta 该层的敏感度图，和输出的维度想同
#endif  // not GPU

	// 卷积层三种指针函数，对应三种计算：前向，反向，更新
    l.forward = forward_convolutional_layer;
    l.backward = backward_convolutional_layer;
    l.update = update_convolutional_layer;
    if(binary){
        l.binary_weights = (float*)xcalloc(l.nweights, sizeof(float));
        l.cweights = (char*)xcalloc(l.nweights, sizeof(char));
        l.scales = (float*)xcalloc(n, sizeof(float));
    }
    if(xnor){
        l.binary_weights = (float*)xcalloc(l.nweights, sizeof(float));
        l.binary_input = (float*)xcalloc(l.inputs * l.batch, sizeof(float));

        int align = 32;// 8;
        int src_align = l.out_h*l.out_w;
        l.bit_align = src_align + (align - src_align % align);

        l.mean_arr = (float*)xcalloc(l.n, sizeof(float));

        const size_t new_c = l.c / 32;
        size_t in_re_packed_input_size = new_c * l.w * l.h + 1;
        l.bin_re_packed_input = (uint32_t*)xcalloc(in_re_packed_input_size, sizeof(uint32_t));

        l.lda_align = 256;  // AVX2
        int k = l.size*l.size*l.c;
        size_t k_aligned = k + (l.lda_align - k%l.lda_align);
        size_t t_bit_input_size = k_aligned * l.bit_align / 8;
        l.t_bit_input = (char*)xcalloc(t_bit_input_size, sizeof(char));
    }

    if(batch_normalize){
        if (l.share_layer) {
            l.scales = l.share_layer->scales;
            l.scale_updates = l.share_layer->scale_updates;
            l.mean = l.share_layer->mean;
            l.variance = l.share_layer->variance;
            l.mean_delta = l.share_layer->mean_delta;
            l.variance_delta = l.share_layer->variance_delta;
            l.rolling_mean = l.share_layer->rolling_mean;
            l.rolling_variance = l.share_layer->rolling_variance;
        }
        else {
            l.scales = (float*)xcalloc(n, sizeof(float));
            for (i = 0; i < n; ++i) {
                l.scales[i] = 1;
            }
            if (train) {
                l.scale_updates = (float*)xcalloc(n, sizeof(float));

                l.mean = (float*)xcalloc(n, sizeof(float));
                l.variance = (float*)xcalloc(n, sizeof(float));

                l.mean_delta = (float*)xcalloc(n, sizeof(float));
                l.variance_delta = (float*)xcalloc(n, sizeof(float));
            }
            l.rolling_mean = (float*)xcalloc(n, sizeof(float));
            l.rolling_variance = (float*)xcalloc(n, sizeof(float));
        }

#ifndef GPU
        if (train) {
            l.x = (float*)xcalloc(total_batch * l.outputs, sizeof(float));
            l.x_norm = (float*)xcalloc(total_batch * l.outputs, sizeof(float));
        }
#endif  // not GPU
    }

#ifndef GPU
    if (l.activation == SWISH || l.activation == MISH) l.activation_input = (float*)calloc(total_batch*l.outputs, sizeof(float));
#endif  // not GPU

    if(adam){
        l.adam = 1;
        l.m = (float*)xcalloc(l.nweights, sizeof(float));
        l.v = (float*)xcalloc(l.nweights, sizeof(float));
        l.bias_m = (float*)xcalloc(n, sizeof(float));
        l.scale_m = (float*)xcalloc(n, sizeof(float));
        l.bias_v = (float*)xcalloc(n, sizeof(float));
        l.scale_v = (float*)xcalloc(n, sizeof(float));
    }

#ifdef GPU


    l.forward_gpu = forward_convolutional_layer_gpu;
    l.backward_gpu = backward_convolutional_layer_gpu;
    l.update_gpu = update_convolutional_layer_gpu;

    if(gpu_index >= 0){

        if (l.activation == SWISH || l.activation == MISH) {
            l.activation_input_gpu = cuda_make_array(l.activation_input, total_batch*l.outputs);
        }

        if (l.deform) l.weight_deform_gpu = cuda_make_array(NULL, l.nweights);

        if (adam) {
            l.m_gpu = cuda_make_array(l.m, l.nweights);
            l.v_gpu = cuda_make_array(l.v, l.nweights);
            l.bias_m_gpu = cuda_make_array(l.bias_m, n);
            l.bias_v_gpu = cuda_make_array(l.bias_v, n);
            l.scale_m_gpu = cuda_make_array(l.scale_m, n);
            l.scale_v_gpu = cuda_make_array(l.scale_v, n);
        }
        if (l.share_layer) {
            l.weights_gpu = l.share_layer->weights_gpu;
            l.weight_updates_gpu = l.share_layer->weight_updates_gpu;
            l.weights_gpu16 = l.share_layer->weights_gpu16;
            l.weight_updates_gpu16 = l.share_layer->weight_updates_gpu16;
            l.biases_gpu = l.share_layer->biases_gpu;
            l.bias_updates_gpu = l.share_layer->bias_updates_gpu;
        }
        else {
            l.weights_gpu = cuda_make_array(l.weights, l.nweights);
            if (train) l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);
#ifdef CUDNN_HALF
            l.weights_gpu16 = cuda_make_array(NULL, l.nweights / 2 + 1);
            if (train) l.weight_updates_gpu16 = cuda_make_array(NULL, l.nweights / 2 + 1);
#endif  // CUDNN_HALF
            l.biases_gpu = cuda_make_array(l.biases, n);
            if (train) l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);
        }

        l.output_gpu = cuda_make_array(l.output, total_batch*out_h*out_w*n);
        if (train) l.delta_gpu = cuda_make_array(l.delta, total_batch*out_h*out_w*n);

        if(binary){
            l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
        }
        if(xnor){
            l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
            l.mean_arr_gpu = cuda_make_array(0, l.n);
            l.binary_input_gpu = cuda_make_array(0, l.inputs*l.batch);
        }

        if(batch_normalize){
            if (l.share_layer) {
                l.scales_gpu = l.share_layer->scales_gpu;
                l.scale_updates_gpu = l.share_layer->scale_updates_gpu;
                l.mean_gpu = l.share_layer->mean_gpu;
                l.variance_gpu = l.share_layer->variance_gpu;
                l.rolling_mean_gpu = l.share_layer->rolling_mean_gpu;
                l.rolling_variance_gpu = l.share_layer->rolling_variance_gpu;
                l.mean_delta_gpu = l.share_layer->mean_delta_gpu;
                l.variance_delta_gpu = l.share_layer->variance_delta_gpu;
            }
            else {
                l.scales_gpu = cuda_make_array(l.scales, n);

                if (train) {
                    l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

                    l.mean_gpu = cuda_make_array(l.mean, n);
                    l.variance_gpu = cuda_make_array(l.variance, n);
#ifndef CUDNN
                    l.mean_delta_gpu = cuda_make_array(l.mean, n);
                    l.variance_delta_gpu = cuda_make_array(l.variance, n);
#endif  // CUDNN
                }

                l.rolling_mean_gpu = cuda_make_array(l.mean, n);
                l.rolling_variance_gpu = cuda_make_array(l.variance, n);
            }

            if (train) {
                l.x_gpu = cuda_make_array(l.output, total_batch*out_h*out_w*n);
#ifndef CUDNN
                l.x_norm_gpu = cuda_make_array(l.output, total_batch*out_h*out_w*n);
#endif  // CUDNN
            }
        }

        if (l.assisted_excitation)
        {
            const int size = l.out_w * l.out_h * l.batch;
            l.gt_gpu = cuda_make_array(NULL, size);
            l.a_avg_gpu = cuda_make_array(NULL, size);
        }
#ifdef CUDNN
        create_convolutional_cudnn_tensors(&l);
        cudnn_convolutional_setup(&l, cudnn_fastest, 0);
#endif  // CUDNN
    }
#endif  // GPU
    l.workspace_size = get_convolutional_workspace_size(l);

    //fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    l.bflops = (2.0 * l.nweights * l.out_h*l.out_w) / 1000000000.;
    if (l.xnor) l.bflops = l.bflops / 32;
    if (l.xnor && l.use_bin_output) fprintf(stderr, "convXB");
    else if (l.xnor) fprintf(stderr, "convX ");
    else if (l.share_layer) fprintf(stderr, "convS ");
    else if (l.assisted_excitation) fprintf(stderr, "convAE");
    else fprintf(stderr, "conv  ");

    if (groups > 1) fprintf(stderr, "%5d/%4d ", n, groups);
    else           fprintf(stderr, "%5d      ", n);

    if (stride_x != stride_y) fprintf(stderr, "%2dx%2d/%2dx%2d ", size, size, stride_x, stride_y);
    else {
        if (dilation > 1) fprintf(stderr, "%2d x%2d/%2d(%1d)", size, size, stride_x, dilation);
        else             fprintf(stderr, "%2d x%2d/%2d   ", size, size, stride_x);
    }

    fprintf(stderr, "%4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", w, h, c, l.out_w, l.out_h, l.out_c, l.bflops);

    //fprintf(stderr, "%5d/%2d %2d x%2d /%2d(%d)%4d x%4d x%4d  -> %4d x%4d x%4d %5.3f BF\n", n, groups, size, size, stride, dilation, w, h, c, l.out_w, l.out_h, l.out_c, l.bflops);

    if (l.antialiasing) {
        printf("AA:  ");
        l.input_layer = (layer*)calloc(1, sizeof(layer));
        int blur_size = 3;
        int blur_pad = blur_size / 2;
        if (l.antialiasing == 2) {
            blur_size = 2;
            blur_pad = 0;
        }
        *(l.input_layer) = make_convolutional_layer(batch, steps, out_h, out_w, n, n, n, blur_size, blur_stride_x, blur_stride_y, 1, blur_pad, LINEAR, 0, 0, 0, 0, 0, index, 0, NULL, 0, 0, train);
        const int blur_nweights = n * blur_size * blur_size;  // (n / n) * n * blur_size * blur_size;
        int i;
        if (blur_size == 2) {
            for (i = 0; i < blur_nweights; i += (blur_size*blur_size)) {
                l.input_layer->weights[i + 0] = 1 / 4.f;
                l.input_layer->weights[i + 1] = 1 / 4.f;
                l.input_layer->weights[i + 2] = 1 / 4.f;
                l.input_layer->weights[i + 3] = 1 / 4.f;
            }
        }
        else {
            for (i = 0; i < blur_nweights; i += (blur_size*blur_size)) {
                l.input_layer->weights[i + 0] = 1 / 16.f;
                l.input_layer->weights[i + 1] = 2 / 16.f;
                l.input_layer->weights[i + 2] = 1 / 16.f;

                l.input_layer->weights[i + 3] = 2 / 16.f;
                l.input_layer->weights[i + 4] = 4 / 16.f;
                l.input_layer->weights[i + 5] = 2 / 16.f;

                l.input_layer->weights[i + 6] = 1 / 16.f;
                l.input_layer->weights[i + 7] = 2 / 16.f;
                l.input_layer->weights[i + 8] = 1 / 16.f;
            }
        }
        for (i = 0; i < n; ++i) l.input_layer->biases[i] = 0;
#ifdef GPU
        if (gpu_index >= 0) {
            l.input_antialiasing_gpu = cuda_make_array(NULL, l.batch*l.outputs);
            push_convolutional_layer(*(l.input_layer));
        }
#endif  // GPU
    }

    return l;
}

void denormalize_convolutional_layer(convolutional_layer l)
{
    int i, j;
    for(i = 0; i < l.n; ++i){
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .00001);
        for(j = 0; j < l.nweights; ++j){
            l.weights[i*l.nweights + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}

void test_convolutional_layer()
{
    convolutional_layer l = make_convolutional_layer(1, 1, 5, 5, 3, 2, 1, 5, 2, 2, 1, 1, LEAKY, 1, 0, 0, 0, 0, 0, 0, NULL, 0, 0, 0);
    l.batch_normalize = 1;
    float data[] = {1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3};
    network_state state = {0};
    state.input = data;
    forward_convolutional_layer(l, state);
}

void resize_convolutional_layer(convolutional_layer *l, int w, int h)
{
    int total_batch = l->batch*l->steps;
    int old_w = l->w;
    int old_h = l->h;
    l->w = w;
    l->h = h;
    int out_w = convolutional_out_width(*l);
    int out_h = convolutional_out_height(*l);

    l->out_w = out_w;
    l->out_h = out_h;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;


    l->output = (float*)xrealloc(l->output, total_batch * l->outputs * sizeof(float));
    if (l->train) {
        l->delta = (float*)xrealloc(l->delta, total_batch * l->outputs * sizeof(float));

        if (l->batch_normalize) {
            l->x = (float*)xrealloc(l->x, total_batch * l->outputs * sizeof(float));
            l->x_norm = (float*)xrealloc(l->x_norm, total_batch * l->outputs * sizeof(float));
        }
    }

    if (l->xnor) {
        //l->binary_input = realloc(l->inputs*l->batch, sizeof(float));
    }

    if (l->activation == SWISH || l->activation == MISH) l->activation_input = (float*)realloc(l->activation_input, total_batch*l->outputs * sizeof(float));
#ifdef GPU
    if (old_w < w || old_h < h) {
        if (l->train) {
            cuda_free(l->delta_gpu);
            l->delta_gpu = cuda_make_array(l->delta, total_batch*l->outputs);
        }

        cuda_free(l->output_gpu);
        l->output_gpu = cuda_make_array(l->output, total_batch*l->outputs);

        if (l->batch_normalize) {
            cuda_free(l->x_gpu);
            l->x_gpu = cuda_make_array(l->output, total_batch*l->outputs);

#ifndef CUDNN
            cuda_free(l->x_norm_gpu);
            l->x_norm_gpu = cuda_make_array(l->output, total_batch*l->outputs);
#endif  // CUDNN
        }

        if (l->xnor) {
            cuda_free(l->binary_input_gpu);
            l->binary_input_gpu = cuda_make_array(0, l->inputs*l->batch);
        }

        if (l->activation == SWISH || l->activation == MISH) {
            cuda_free(l->activation_input_gpu);
            l->activation_input_gpu = cuda_make_array(l->activation_input, total_batch*l->outputs);
        }

        if (l->assisted_excitation)
        {
            cuda_free(l->gt_gpu);
            cuda_free(l->a_avg_gpu);

            const int size = l->out_w * l->out_h * l->batch;
            l->gt_gpu = cuda_make_array(NULL, size);
            l->a_avg_gpu = cuda_make_array(NULL, size);
        }
    }
#ifdef CUDNN
    cudnn_convolutional_setup(l, cudnn_fastest, 0);
#endif
#endif
    l->workspace_size = get_convolutional_workspace_size(*l);

#ifdef CUDNN
    // check for excessive memory consumption
    size_t free_byte;
    size_t total_byte;
    CHECK_CUDA(cudaMemGetInfo(&free_byte, &total_byte));
    if (l->workspace_size > free_byte || l->workspace_size >= total_byte / 2) {
        printf(" used slow CUDNN algo without Workspace! Need memory: %zu, available: %zu\n", l->workspace_size, (free_byte < total_byte/2) ? free_byte : total_byte/2);
        cudnn_convolutional_setup(l, cudnn_smallest, 0);
        l->workspace_size = get_convolutional_workspace_size(*l);
    }
#endif
}

void set_specified_workspace_limit(convolutional_layer *l, size_t workspace_size_limit)
{
#ifdef CUDNN
    size_t free_byte;
    size_t total_byte;
    CHECK_CUDA(cudaMemGetInfo(&free_byte, &total_byte));
    cudnn_convolutional_setup(l, cudnn_specify, workspace_size_limit);
    l->workspace_size = get_convolutional_workspace_size(*l);
    //printf("Set specified workspace limit for cuDNN: %zu, available: %zu, workspace = %zu \n", workspace_size_limit, free_byte, l->workspace_size);
#endif  // CUDNN
}

void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void scale_bias(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

/*
** 计算每个卷积核的偏置更新值，所谓偏置更新值，就是bias = bias - alpha * bias_update中的bias_update
** 输入： bias_updates     当前层所有偏置的更新值，维度为l.n（即当前层卷积核的个数）
**       delta            当前层的敏感度图（即l.delta）
**       batch            一个batch含有的图片张数（即l.batch）
**       n                当前层卷积核个数（即l.n）
**       k                当前层输入特征图尺寸（即l.out_w*l.out_h）
** 原理：当前层的敏感度图l.delta是误差函数对加权输入的导数，也就是偏置更新值，只是其中每l.out_w*l.out_h个元素都对应同一个
**      偏置，因此需要将其加起来，得到的和就是误差函数对当前层各偏置的导数（l.delta的维度为l.batch*l.n*l.out_h*l.out_w,
**      可理解成共有l.batch行，每行有l.n*l.out_h*l.out_w列，而这一大行又可以理解成有l.n，l.out_h*l.out_w列，这每一小行就
**      对应同一个卷积核也即同一个偏置）
*/
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
	// 遍历batch中每张输入图片
    // 注意，最后的偏置更新值是所有输入图片的总和（多张图片无非就是重复一张图片的操作，求和即可）。
    // 总之：一个卷积核对应一个偏置更新值，该偏置更新值等于batch中所有输入图片累积的偏置更新值，
    // 而每张图片也需要进行偏置更新值求和（因为每个卷积核在每张图片多个位置做了卷积运算，这都对偏置更新值有贡献）以得到每张图片的总偏置更新值。
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}

void gemm_nn_custom(int M, int N, int K, float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float *C, int ldc)
{
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            PUT_IN_REGISTER float A_PART = ALPHA * A[i * lda + k];
            //printf("\n weight = %f \n", A_PART);
            for (j = 0; j < N; ++j) {
                C[i*ldc + j] += A_PART*B[k*ldb + j];
            }
        }
    }
}


void get_mean_array(float *src, size_t size, size_t filters, float *mean_arr) {
    size_t i, counter;
    counter = 0;
    for (i = 0; i < size; i += size / filters) {
        mean_arr[counter++] = fabs(src[i]);
    }
}

/*
void float_to_bit(float *src, unsigned char *dst, size_t size) {

    size_t dst_size = size / 8 + 1;
    memset(dst, 0, dst_size);
    size_t i, dst_i, dst_shift;
    for (i = 0; i < size; ++i) {
        if (src[i] > 0) set_bit(dst, i);
    }
}
*/

void bit_to_float(unsigned char *src, float *dst, size_t size, size_t filters, float *mean_arr) {
    memset(dst, 0, size *sizeof(float));
    size_t i;

    for (i = 0; i < size; ++i) {
        float mean_val = 1;
        if(mean_arr != NULL) mean_val = fabs(mean_arr[i / (size / filters)]);
        if(get_bit(src, i)) dst[i] = mean_val;
        else dst[i] = -mean_val;
    }
}

void binary_align_weights(convolutional_layer *l)
{
    int m = l->n;   // (l->n / l->groups)
    int k = l->size*l->size*l->c;   // ->size*l->size*(l->c / l->groups)
    size_t new_lda = k + (l->lda_align - k % l->lda_align); // (k / 8 + 1) * 8;
    l->new_lda = new_lda;

    binarize_weights(l->weights, m, k, l->binary_weights);

    size_t align_weights_size = new_lda * m;
    l->align_bit_weights_size = align_weights_size / 8 + 1;
    float* align_weights = (float*)xcalloc(align_weights_size, sizeof(float));
    l->align_bit_weights = (char*)xcalloc(l->align_bit_weights_size, sizeof(char));

    size_t i, j;
    // align A without transpose
    for (i = 0; i < m; ++i) {
        for (j = 0; j < k; ++j) {
            align_weights[i*new_lda + j] = l->binary_weights[i*k + j];
        }
    }


    if (l->c % 32 == 0)
    //if(gpu_index < 0 && l->stride == 1 && l->pad == 1 && l->c % 32 == 0)
    //if (l->stride == 1 && l->pad == 1 && l->c % 32 == 0)
    {
        int fil, chan;
        const int items_per_filter = l->c * l->size * l->size;
        //const int dst_items_per_filter = new_lda;
        for (fil = 0; fil < l->n; ++fil)
        {
            for (chan = 0; chan < l->c; chan += 32)
            {
                const int items_per_channel = l->size*l->size;
                for (i = 0; i < items_per_channel; ++i)
                {
                    //uint32_t val = 0;
                    int c_pack;
                    for (c_pack = 0; c_pack < 32; ++c_pack) {
                        float src = l->binary_weights[fil*items_per_filter + (chan + c_pack)*items_per_channel + i];

                        //align_weights[fil*items_per_filter + chan*items_per_channel + i * 32 + c_pack] = src;

                        align_weights[fil*new_lda + chan*items_per_channel + i*32 + c_pack] = src;
                        //val |= (src << c);
                    }

                }
            }
        }

        //printf("\n l.index = %d \t aw[0] = %f, aw[1] = %f, aw[2] = %f, aw[3] = %f \n", l->index, align_weights[0], align_weights[1], align_weights[2], align_weights[3]);
        //memcpy(l->binary_weights, align_weights, (l->size * l->size * l->c * l->n) * sizeof(float));

        float_to_bit(align_weights, (unsigned char*)l->align_bit_weights, align_weights_size);

        //if (l->n >= 32)
        if(gpu_index >= 0)
        {
            //int M = l->n;
            //int N = l->out_w*l->out_h;
            //printf("\n M = %d, N = %d, M %% 8 = %d, N %% 8 = %d - weights \n", M, N, M % 8, N % 8);
            //printf("\n l.w = %d, l.c = %d, l.n = %d \n", l->w, l->c, l->n);
            for (i = 0; i < align_weights_size / 8; ++i) l->align_bit_weights[i] = ~(l->align_bit_weights[i]);
        }



        get_mean_array(l->binary_weights, m*k, l->n, l->mean_arr);
        //get_mean_array(l->binary_weights, m*new_lda, l->n, l->mean_arr);
    }
    else {
        float_to_bit(align_weights, (unsigned char*)l->align_bit_weights, align_weights_size);

        get_mean_array(l->binary_weights, m*k, l->n, l->mean_arr);
    }

    //l->mean_arr = calloc(l->n, sizeof(float));

    //get_mean_array(align_weights, align_weights_size, l->n, l->mean_arr);




#ifdef GPU
    cudaError_t status;
    l->align_workspace_size = l->bit_align * l->size * l->size * l->c;
    status = cudaMalloc((void **)&l->align_workspace_gpu, l->align_workspace_size * sizeof(float));
    status = cudaMalloc((void **)&l->transposed_align_workspace_gpu, l->align_workspace_size * sizeof(float));
    CHECK_CUDA(status);

    //l->align_bit_weights_gpu = cuda_make_array(l->align_bit_weights, l->align_bit_weights_size * sizeof(char)/sizeof(float));
    status = cudaMalloc((void **)&l->align_bit_weights_gpu, l->align_bit_weights_size);
    CHECK_CUDA(status);
    status = cudaMemcpy(l->align_bit_weights_gpu, l->align_bit_weights, l->align_bit_weights_size, cudaMemcpyHostToDevice);
    CHECK_CUDA(status);
    status = cudaMemcpy(l->binary_weights_gpu, l->binary_weights, m*k * sizeof(float), cudaMemcpyHostToDevice);
    CHECK_CUDA(status);

    //l->mean_arr_gpu = cuda_make_array(l->mean_arr, l->n);
    cuda_push_array(l->mean_arr_gpu, l->mean_arr, l->n);
    CHECK_CUDA(cudaDeviceSynchronize());
#endif // GPU

    free(align_weights);
}

// binary transpose
size_t binary_transpose_align_input(int k, int n, float *b, char **t_bit_input, size_t ldb_align, int bit_align)
{
    size_t new_ldb = k + (ldb_align - k%ldb_align); // (k / 8 + 1) * 8;
    //printf("\n n = %d, bit_align = %d \n", n, bit_align);
    size_t t_intput_size = new_ldb * bit_align;// n;
    size_t t_bit_input_size = t_intput_size / 8;// +1;

    memset(*t_bit_input, 0, t_bit_input_size * sizeof(char));
    //int src_size = k * bit_align;

    // b - [bit_align, k] - [l.bit_align, l.size*l.size*l.c] = src_size
    // t_input - [bit_align, k] - [n', k]
    // t_bit_input - [new_ldb, n] - [k', n]

    //transpose_bin(t_input, *t_bit_input, k, n, bit_align, new_ldb, 8);
    transpose_bin((uint32_t*)b, (uint32_t*)*t_bit_input, k, n, bit_align, new_ldb, 8);

    return t_intput_size;
}


// 卷积层的前向传播核心代码
void forward_convolutional_layer(convolutional_layer l, network_state state)
{
    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
    int i, j;
	// l.outputs = l.out_h * l.out_w * l.out_c在make各网络层函数中赋值（比如make_convolutional_layer()），
    // 对应每张输入图片的所有输出特征图的总元素个数（每张输入图片会得到n也即l.out_c张特征图）
    // 初始化输出l.output全为0.0；输入l.outputs*l.batch为输出的总元素个数，其中l.outputs为batch
    // 中一个输入对应的输出的所有元素的个数，l.batch为一个batch输入包含的图片张数；0表示初始化所有输出为0；
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

	// 是否进行二值化操作
    if (l.xnor && (!l.align_bit_weights || state.train)) {
        if (!l.align_bit_weights || state.train) {
            binarize_weights(l.weights, l.n, l.nweights, l.binary_weights);
            //printf("\n binarize_weights l.align_bit_weights = %p \n", l.align_bit_weights);
        }
        swap_binary(&l);
        binarize_cpu(state.input, l.c*l.h*l.w*l.batch, l.binary_input);
        state.input = l.binary_input;
    }

    int m = l.n / l.groups; // 该层的卷积核个数
    int k = l.size*l.size*l.c / l.groups; // 该层每个卷积核的参数元素个数
    int n = out_h*out_w; // 该层每个特征图的尺寸(元素个数)

    static int u = 0;
    u++;
    // 该循环即为卷积计算核心代码：所有卷积核对batch中每张图片进行卷积运算
    // 每次循环处理一张输入图片（所有卷积核对batch中一张图片做卷积运算）
    for(i = 0; i < l.batch; ++i)
    {
		// 该循环是为了处理分组卷积
        for (j = 0; j < l.groups; ++j)
        {
			// 当前组卷积核(也即权重)，元素个数为l.n*l.c/l.groups*l.size*l.size,
            // 共有l.n行，l.c/l.gropus,l.c*l.size*l.size列
            float *a = l.weights +j*l.nweights / l.groups;
			// 对输入图像进行重排之后的图像数据，所以内存空间申请为网络中最大占用内存
            float *b = state.workspace;
			// 存储一张输入图片（多通道）当前组的输出特征图（输入图片是多通道的，输出
            // 图片也是多通道的，有多少组卷积核就有多少组通道，每个分组后的卷积核得到一张特征图即为一个通道）
            // 这里似乎有点拗口，可以看下分组卷积原理。
            float *c = l.output +(i*l.groups + j)*n*m;

            //gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
            //gemm_nn_custom(m, n, k, 1, a, k, b, n, c, n);
			//二值网络，特殊处理，里面还有一些优化，细节很多，这里暂时不管二值网络这部分，把注意力先放在普通卷积层的计算上
            if (l.xnor && l.align_bit_weights && !state.train && l.stride_x == l.stride_y)
            {
                memset(b, 0, l.bit_align*l.size*l.size*l.c * sizeof(float));

                if (l.c % 32 == 0)
                {
                    //printf(" l.index = %d - new XNOR \n", l.index);

                    int ldb_align = l.lda_align;
                    size_t new_ldb = k + (ldb_align - k%ldb_align); // (k / 8 + 1) * 8;
                    //size_t t_intput_size = new_ldb * l.bit_align;// n;
                    //size_t t_bit_input_size = t_intput_size / 8;// +1;

                    int re_packed_input_size = l.c * l.w * l.h;
                    memset(state.workspace, 0, re_packed_input_size * sizeof(float));

                    const size_t new_c = l.c / 32;
                    size_t in_re_packed_input_size = new_c * l.w * l.h + 1;
                    memset(l.bin_re_packed_input, 0, in_re_packed_input_size * sizeof(uint32_t));

                    //float *re_packed_input = calloc(l.c * l.w * l.h, sizeof(float));
                    //uint32_t *bin_re_packed_input = calloc(new_c * l.w * l.h + 1, sizeof(uint32_t));

                    // float32x4 by channel (as in cuDNN)
                    repack_input(state.input, state.workspace, l.w, l.h, l.c);

                    // 32 x floats -> 1 x uint32_t
                    float_to_bit(state.workspace, (unsigned char *)l.bin_re_packed_input, l.c * l.w * l.h);

                    //free(re_packed_input);

                    // slow - convolution the packed inputs and weights: float x 32 by channel (as in cuDNN)
                    //convolution_repacked((uint32_t *)bin_re_packed_input, (uint32_t *)l.align_bit_weights, l.output,
                    //    l.w, l.h, l.c, l.n, l.size, l.pad, l.new_lda, l.mean_arr);

                    // // then exit from if()


                    im2col_cpu_custom((float *)l.bin_re_packed_input, new_c, l.h, l.w, l.size, l.stride, l.pad, state.workspace);
                    //im2col_cpu((float *)bin_re_packed_input, new_c, l.h, l.w, l.size, l.stride, l.pad, b);

                    //free(bin_re_packed_input);

                    int new_k = l.size*l.size*l.c / 32;

                    // good for (l.c == 64)
                    //gemm_nn_bin_32bit_packed(m, n, new_k, 1,
                    //    l.align_bit_weights, l.new_lda/32,
                    //    b, n,
                    //    c, n, l.mean_arr);

    // // then exit from if()

                    transpose_uint32((uint32_t *)state.workspace, (uint32_t*)l.t_bit_input, new_k, n, n, new_ldb);

                    // the main GEMM function
                    gemm_nn_custom_bin_mean_transposed(m, n, k, 1, (unsigned char*)l.align_bit_weights, new_ldb, (unsigned char*)l.t_bit_input, new_ldb, c, n, l.mean_arr);

                    // // alternative GEMM
                    //gemm_nn_bin_transposed_32bit_packed(m, n, new_k, 1,
                    //    l.align_bit_weights, l.new_lda/32,
                    //    t_bit_input, new_ldb / 32,
                    //    c, n, l.mean_arr);

                    //free(t_bit_input);

                }
                else
                { // else (l.c % 32 != 0)

                    //--------------------------------------------------------
                    //printf(" l.index = %d - old XNOR \n", l.index);

                    //im2col_cpu_custom_align(state.input, l.c, l.h, l.w, l.size, l.stride, l.pad, b, l.bit_align);
                    im2col_cpu_custom_bin(state.input, l.c, l.h, l.w, l.size, l.stride, l.pad, state.workspace, l.bit_align);

                    //size_t output_size = l.outputs;
                    //float *count_output = calloc(output_size, sizeof(float));
                    //size_t bit_output_size = output_size / 8 + 1;
                    //char *bit_output = calloc(bit_output_size, sizeof(char));

                    //size_t intput_size = n * k; // (out_h*out_w) X (l.size*l.size*l.c) : after im2col()
                    //size_t bit_input_size = intput_size / 8 + 1;
                    //char *bit_input = calloc(bit_input_size, sizeof(char));

                    //size_t weights_size = k * m; //l.size*l.size*l.c*l.n; // l.nweights
                    //size_t bit_weights_size = weights_size / 8 + 1;

                    //char *bit_weights = calloc(bit_weights_size, sizeof(char));
                    //float *mean_arr = calloc(l.n, sizeof(float));

                    // transpose B from NxK to KxN (x-axis (ldb = l.size*l.size*l.c) - should be multiple of 8 bits)
                    {
                        //size_t ldb_align = 256; // 256 bit for AVX2
                        int ldb_align = l.lda_align;
                        size_t new_ldb = k + (ldb_align - k%ldb_align);
                        size_t t_intput_size = binary_transpose_align_input(k, n, state.workspace, &l.t_bit_input, ldb_align, l.bit_align);

                        // 5x times faster than gemm()-float32
                        gemm_nn_custom_bin_mean_transposed(m, n, k, 1, (unsigned char*)l.align_bit_weights, new_ldb, (unsigned char*)l.t_bit_input, new_ldb, c, n, l.mean_arr);

                        //gemm_nn_custom_bin_mean_transposed(m, n, k, 1, bit_weights, k, t_bit_input, new_ldb, c, n, mean_arr);

                        //free(t_input);
                        //free(t_bit_input);
                        //}
                    }

                }

                add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);

                //activate_array(l.output, m*n*l.batch, l.activation);
                if (l.activation == SWISH) activate_array_swish(l.output, l.outputs*l.batch, l.activation_input, l.output);
                else if (l.activation == MISH) activate_array_mish(l.output, l.outputs*l.batch, l.activation_input, l.output);
                else if (l.activation == NORM_CHAN) activate_array_normalize_channels(l.output, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.output);
                else if (l.activation == NORM_CHAN_SOFTMAX) activate_array_normalize_channels_softmax(l.output, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.output, 0);
                else if (l.activation == NORM_CHAN_SOFTMAX_MAXVAL) activate_array_normalize_channels_softmax(l.output, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.output, 1);
                else activate_array_cpu_custom(l.output, m*n*l.batch, l.activation);
                return;

            }
            else {
                //printf(" l.index = %d - FP32 \n", l.index);
				// 由于有分组卷积，所以获取属于当前组的输入im并按一定存储规则排列的数组b，
				// 以方便、高效地进行矩阵（卷积）计算，详细查看该函数注释（比较复杂）
				// 这里的im实际上只加载了一张图片的数据
				//关于im2col的原理我会讲
                float *im = state.input + (i*l.groups + j)*(l.c / l.groups)*l.h*l.w;
				// 如果这里卷积核尺寸为1，是不需要改变内存排布方式
                if (l.size == 1) {
                    b = im;
                }
                else {
                    //im2col_cpu(im, l.c / l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
					// 将多通道二维图像im变成按一定存储规则排列的数组b，
					// 以方便、高效地进行矩阵（卷积）计算，详细查看该函数注释（比较复杂）
					// 进行重排，l.c/groups为每张图片的通道数分组，l.h为每张图片的高度，l.w为每张图片的宽度，l.size为卷积核尺寸，l.stride为步长
					// 得到的b为一张图片重排后的结果，也是按行存储的一维数组（共有l.c/l.groups*l.size*l.size行，l.out_w*l.out_h列）
                    im2col_cpu_ext(im,   // input
                        l.c / l.groups,     // input channels
                        l.h, l.w,           // input size (h, w)
                        l.size, l.size,     // kernel size (h, w)
                        l.pad, l.pad,       // padding (h, w)
                        l.stride_y, l.stride_x, // stride (h, w)
                        l.dilation, l.dilation, // dilation (h, w)
                        b);                 // output

                }
				// 此处在im2col_cpu操作基础上，利用矩阵乘法c=alpha*a*b+beta*c完成对图像卷积的操作
				// 0,0表示不对输入a,b进行转置，
				// m是输入a,c的行数，具体含义为每个卷积核的个数，
				// n是输入b,c的列数，具体含义为每个输出特征图的元素个数(out_h*out_w)，
				// k是输入a的列数也是b的行数，具体含义为卷积核元素个数乘以输入图像的通道数除以分组数（l.size*l.size*l.c/l.groups），
				// a,b,c即为三个参与运算的矩阵（用一维数组存储）,alpha=beta=1为常系数，
				// a为所有卷积核集合,元素个数为l.n*l.c/l.groups*l.size*l.size，按行存储，共有l*n行，l.c/l.groups*l.size*l.size列，
				// 即a中每行代表一个可以作用在3通道上的卷积核，
				// b为一张输入图像经过im2col_cpu重排后的图像数据（共有l.c/l.group*l.size*l.size行，l.out_w*l.out_h列），
				// c为gemm()计算得到的值，包含一张输入图片得到的所有输出特征图（每个卷积核得到一张特征图），c中一行代表一张特征图，
				// 各特征图铺排开成一行后，再将所有特征图并成一大行，存储在c中，因此c可视作有l.n行，l.out_h*l.out_w列。
				// 详细查看该函数注释（比较复杂）
                gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
                // bit-count to float
            }
            //c += n*m;
            //state.input += l.c*l.h*l.w;
        }
    }
	// 如果卷积层使用了BatchNorm，那么执行forward_batchnorm，如果没有，则添加偏置
    if(l.batch_normalize){
        forward_batchnorm_layer(l, state);
    }
    else {
        add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);
    }

    //activate_array(l.output, m*n*l.batch, l.activation);
	// 使用不同的激活函数
    if (l.activation == SWISH) activate_array_swish(l.output, l.outputs*l.batch, l.activation_input, l.output);
    else if (l.activation == MISH) activate_array_mish(l.output, l.outputs*l.batch, l.activation_input, l.output);
    else if (l.activation == NORM_CHAN) activate_array_normalize_channels(l.output, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.output);
    else if (l.activation == NORM_CHAN_SOFTMAX) activate_array_normalize_channels_softmax(l.output, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.output, 0);
    else if (l.activation == NORM_CHAN_SOFTMAX_MAXVAL) activate_array_normalize_channels_softmax(l.output, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.output, 1);
    else activate_array_cpu_custom(l.output, l.outputs*l.batch, l.activation);
	// 二值网络，前向传播结束之后转回float
    if(l.binary || l.xnor) swap_binary(&l);

    //visualize_convolutional_layer(l, "conv_visual", NULL);
    //wait_until_press_key_cv();
	// 暂时不懂
    if(l.assisted_excitation && state.train) assisted_excitation_forward(l, state);
	// 暂时不懂
    if (l.antialiasing) {
        network_state s = { 0 };
        s.train = state.train;
        s.workspace = state.workspace;
        s.net = state.net;
        s.input = l.output;
        forward_convolutional_layer(*(l.input_layer), s);
        //simple_copy_ongpu(l.outputs*l.batch, l.output, l.input_antialiasing);
        memcpy(l.output, l.input_layer->output, l.input_layer->outputs * l.input_layer->batch * sizeof(float));
    }
}

void assisted_excitation_forward(convolutional_layer l, network_state state)
{
    const int iteration_num = (*state.net.seen) / (state.net.batch*state.net.subdivisions);

    // epoch
    //const float epoch = (float)(*state.net.seen) / state.net.train_images_num;

    // calculate alpha
    //const float alpha = (1 + cos(3.141592 * iteration_num)) / (2 * state.net.max_batches);
    //const float alpha = (1 + cos(3.141592 * epoch)) / (2 * state.net.max_batches);
    float alpha = (1 + cos(3.141592 * iteration_num / state.net.max_batches));

    if (l.assisted_excitation > 1) {
        if (iteration_num > l.assisted_excitation) alpha = 0;
        else alpha = (1 + cos(3.141592 * iteration_num / l.assisted_excitation));
    }

    //printf("\n epoch = %f, alpha = %f, seen = %d, max_batches = %d, train_images_num = %d \n",
    //    epoch, alpha, (*state.net.seen), state.net.max_batches, state.net.train_images_num);

    float *a_avg = (float *)xcalloc(l.out_w * l.out_h * l.batch, sizeof(float));
    float *g = (float *)xcalloc(l.out_w * l.out_h * l.batch, sizeof(float));

    int b;
    int w, h, c;

    l.max_boxes = state.net.num_boxes;
    l.truths = l.max_boxes*(4 + 1);

    for (b = 0; b < l.batch; ++b)
    {
        // calculate G
        int t;
        for (t = 0; t < state.net.num_boxes; ++t) {
            box truth = float_to_box_stride(state.truth + t*(4 + 1) + b*l.truths, 1);
            if (!truth.x) break;  // continue;

            int left = floor((truth.x - truth.w / 2) * l.out_w);
            int right = ceil((truth.x + truth.w / 2) * l.out_w);
            int top = floor((truth.y - truth.h / 2) * l.out_h);
            int bottom = ceil((truth.y + truth.h / 2) * l.out_h);

            for (w = left; w <= right; w++) {
                for (h = top; h < bottom; h++) {
                    g[w + l.out_w * h + l.out_w*l.out_h*b] = 1;
                }
            }
        }
    }

    for (b = 0; b < l.batch; ++b)
    {
        // calculate average A
        for (w = 0; w < l.out_w; w++) {
            for (h = 0; h < l.out_h; h++) {
                for (c = 0; c < l.out_c; c++) {
                    a_avg[w + l.out_w*(h + l.out_h*b)] += l.output[w + l.out_w*(h + l.out_h*(c + l.out_c*b))];
                }
                a_avg[w + l.out_w*(h + l.out_h*b)] /= l.out_c;  // a_avg / d
            }
        }
    }

    // change activation
    for (b = 0; b < l.batch; ++b)
    {
        for (w = 0; w < l.out_w; w++) {
            for (h = 0; h < l.out_h; h++) {
                for (c = 0; c < l.out_c; c++)
                {
                    // a = a + alpha(t) + e(c,i,j) = a + alpha(t) + g(i,j) * avg_a(i,j) / channels
                    l.output[w + l.out_w*(h + l.out_h*(c + l.out_c*b))] +=
                        alpha *
                        g[w + l.out_w*(h + l.out_h*b)] *
                        a_avg[w + l.out_w*(h + l.out_h*b)];

                    //l.output[w + l.out_w*(h + l.out_h*(c + l.out_c*b))] =
                    //    alpha * g[w + l.out_w*(h + l.out_h*b)] * a_avg[w + l.out_w*(h + l.out_h*b)];
                }
            }
        }
    }

    if(0)   // visualize ground truth
    {
#ifdef OPENCV
        for (b = 0; b < l.batch; ++b)
        {
            image img = float_to_image(l.out_w, l.out_h, 1, &g[l.out_w*l.out_h*b]);
            char buff[100];
            sprintf(buff, "a_excitation_%d", b);
            show_image_cv(img, buff);

            image img2 = float_to_image(l.out_w, l.out_h, 1, &l.output[l.out_w*l.out_h*l.out_c*b]);
            char buff2[100];
            sprintf(buff2, "a_excitation_act_%d", b);
            show_image_cv(img2, buff2);
            wait_key_cv(5);
        }
        wait_until_press_key_cv();
#endif // OPENCV
    }

    free(g);
    free(a_avg);
}

/*
** 卷积神经网络反向传播核心函数
** 主要流程：1） 调用gradient_array()计算当前层l所有输出元素关于加权输入的导数值（也即激活函数关于输入的导数值），
**             并乘上上一次调用backward_convolutional_layer()还没计算完的l.delta，得到当前层最终的敏感度图；
**          2） 如果网络进行了BN，则backward_batchnorm_layer。
**          3） 如果网络没有进行BN，则直接调用 backward_bias()计算当前层所有卷积核的偏置更新值；
**          4） 依次调用im2col_cpu()，gemm_nt()函数计算当前层权重系数更新值；
**          5） 如果上一层的delta已经动态分配了内存，则依次调用gemm_tn(), col2im_cpu()计算上一层的敏感度图（并未完成所有计算，还差一个步骤）；
** 强调：每次调用本函数会计算完成当前层的敏感度计算，同时计算当前层的偏置、权重更新值，除此之外，还会计算上一层的敏感度图，但是要注意的是，
**      并没有完全计算完，还差一步：乘上激活函数对加权输入的导数值。这一步在下一次调用本函数时完成。
*/
void backward_convolutional_layer(convolutional_layer l, network_state state)
{
    int i, j;
	// 卷积核个数，考虑到分组卷积
    int m = l.n / l.groups; 
	// 每一个卷积核元素个数（包括l.c（l.c为该层网络接受的输入图片的通道数）个通道上的卷积核元素个数总数，比如卷积核尺寸为3*3,
    // 输入图片有3个通道，因为要同时作用于输入的3个通道上，所以实际上这个卷积核是一个立体的，共有3*3*3=27个元素，这些元素都是要训练的参数），同样需要考虑分组数
    int n = l.size*l.size*l.c / l.groups;
	// 每张输出特征图的元素个数：out_w，out_h是输出特征图的宽高
    int k = l.out_w*l.out_h;

	// 计算当前层激活函数对加权输入的导数值并乘以l.delta相应元素，从而彻底完成当前层敏感度图的计算，得到当前层的敏感度图l.delta。
    // l.output存储了该层网络的所有输出：该层网络接受一个batch的输入图片，其中每张图片经卷积处理后得到的特征图尺寸为：l.out_w,l.out_h，
    // 该层卷积网络共有l.n个卷积核，因此一张输入图片共输出l.n张宽高为l.out_w,l.out_h的特征图（l.output为一张图所有输出特征图的总元素个数），
    // 所以所有输入图片也即l.output中的总元素个数为：l.n*l.out_w*l.out_h*l.batch；
    // l.activation为该卷积层的激活函数类型，l.delta就是gradient_array()函数计算得到的l.output中每一个元素关于激活函数函数输入的导数值，
    // 注意，这里直接利用输出值求得激活函数关于输入的导数值是因为神经网络中所使用的绝大部分激活函数关于输入的导数值都可以描述为输出值的函数表达式，
    // 比如对于Sigmoid激活函数（记作f(x)），其导数值为f(x)'=f(x)*(1-f(x)),因此如果给出y=f(x)，那么f(x)'=y*(1-y)，只需要输出值y就可以了，不需要输入x的值，
    // （暂时不确定darknet中有没有使用特殊的激活函数，以致于必须要输入值才能够求出导数值，在activiation.c文件中，有几个激活函数暂时没看懂，也没在网上查到）。
    // l.delta是一个一维数组，长度为l.batch * l.outputs（其中l.outputs = l.out_h * l.out_w * l.out_c），在make_convolutional_layer()动态分配内存；
    // 再强调一次：gradient_array()不单单是完成激活函数对输入的求导运算，还完成计算当前层敏感度图的最后一步：l.delta中每个元素乘以激活函数对输入的导数（注意gradient_arry中使用的是*=运算符）。
    // 每次调用backward_convolutional_laye时，都会完成当前层敏感度图的计算，同时会计算上一层的敏感度图，但对于上一层，其敏感度图并没有完全计算完成，还差一步，
    // 需要等到下一次调用backward_convolutional_layer()时来完成，诚如col2im_cpu()中注释一样。
    if (l.activation == SWISH) gradient_array_swish(l.output, l.outputs*l.batch, l.activation_input, l.delta);
    else if (l.activation == MISH) gradient_array_mish(l.outputs*l.batch, l.activation_input, l.delta);
    else if (l.activation == NORM_CHAN_SOFTMAX || l.activation == NORM_CHAN_SOFTMAX_MAXVAL) gradient_array_normalize_channels_softmax(l.output, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.delta);
    else if (l.activation == NORM_CHAN) gradient_array_normalize_channels(l.output, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.delta);
    else gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

    if (l.batch_normalize) {
		// 之后单独讲BN层的前向和反向传播
        backward_batchnorm_layer(l, state);
    }
    else {
		// 计算偏置的更新值：每个卷积核都有一个偏置，偏置的更新值也即误差函数对偏置的导数，这个导数的计算很简单，实际所有的导数已经求完了，都存储在l.delta中，
        // 接下来只需把l.delta中对应同一个卷积核的项加起来就可以（卷积核在图像上逐行逐列跨步移动做卷积，每个位置处都有一个输出，共有l.out_w*l.out_h个，
        // 这些输出都与同一个偏置关联，因此将l.delta中对应同一个卷积核的项加起来即得误差函数对这个偏置的导数）
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
    }
	// 遍历batch中的每张照片，对于l.delta来说，每张照片是分开存的，因此其维度会达到：l.batch*l.n*l.out_w*l.out_h，
    // 对于l.weights,l.weight_updates以及上面提到的l.bias,l.bias_updates，是将所有照片对应元素叠加起来
    // （循环的过程就是叠加的过程，注意gemm()这系列函数含有叠加效果，不是覆盖输入C的值，而是叠加到之前的C上），
    // 因此l.weights与l.weight_updates维度为l.n*l.size*l.size，l.bias与l.bias_updates的维度为l.h，都与l.batch无关
    for (i = 0; i < l.batch; ++i) {
        for (j = 0; j < l.groups; ++j) {
			float *a = l.delta + (i*l.groups + j)*m*k;
			// net.workspace的元素个数为所有层中最大的l.workspace_size（在make_convolutional_layer()计算得到workspace_size的大小，在parse_network_cfg()中动态分配内存，此值对应未使用gpu时的情况）,
			// net.workspace充当一个临时工作空间的作用，存储临时所需要的计算参数，比如每层单张图片重排后的结果（这些参数马上就会参与卷积运算），一旦用完，就会被马上更新（因此该变量的值的更新频率比较大）
            float *b = state.workspace;
			
            float *c = l.weight_updates + j*l.nweights / l.groups;
			// 进入本函数之前，在backward_network()函数中，已经将net.input赋值为prev.output，也即若当前层为第l层，net.input此时已经是第l-1层的输出
			// 注意反向传播从后往前来看
            float *im = state.input + (i*l.groups + j)* (l.c / l.groups)*l.h*l.w;

            //im2col_cpu(im, l.c / l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
			// 下面两步：im2col_cpu()与gemm()是为了计算当前层的权重更新值（其实也就是误差函数对当前层权重的导数）
			// 将多通道二维图像net.input变成按一定存储规则排列的数组b，以方便、高效地进行矩阵（卷积）计算，详细查看该函数注释（比较复杂），
			// im2col_cpu_ext每次仅处理net.input（包含整个batch）中的一张输入图片（对于第一层，则就是读入的图片，对于之后的层，这些图片都是上一层的输出，通道数等于上一层卷积核个数）。
			// 最终重排的b为l.c * l.size * l.size行，l.out_h * l.out_w列。
			// 你会发现在前向forward_convolutional_layer()函数中，也为每层的输入进行了重排，但是很遗憾的是，并没有一个l.workspace把每一层的重排结果保存下来，而是统一存储到net.workspace中，
			// 并被不断擦除更新，那为什么不保存呢？保存下来不是省掉一大笔额外重复计算开销？原因有两个：1）net.workspace中只存储了一张输入图片的重排结果，所以重排下张图片时，马上就会被擦除，
			// 当然你可能会想，那为什么不弄一个l.worspaces将每层所有输入图片的结果保存呢？这引出第二个原因；2）计算成本是降低了，但存储空间需求急剧增加，想想每一层都有l.batch张图，且每张都是多通道的，
			// 重排后其元素个数还会增多，这个存储量搁谁都受不了，如果一个batch有128张图，输入图片尺寸为400*400，3通道，网络有16层（假设每层输入输出尺寸及通道数都一样），那么单单为了存储这些重排结果，
			// 就需要128*400*400*3*16*4/1024/1024/1024 = 3.66G，所以为了权衡，只能重复计算！
            im2col_cpu_ext(
                im,                 // input
                l.c / l.groups,     // input channels
                l.h, l.w,           // input size (h, w)
                l.size, l.size,     // kernel size (h, w)
                l.pad, l.pad,       // padding (h, w)
                l.stride_y, l.stride_x, // stride (h, w)
                l.dilation, l.dilation, // dilation (h, w)
                b);                 // output
			// 下面计算当前层的权重更新值，所谓权重更新值就是weight = weight - alpha * weight_update中的weight_update，
			// 权重更新值等于当前层敏感度图中每个元素乘以相应的像素值，因为一个权重跟当前层多个输出有关联（权值共享，即卷积核在图像中跨步移动做卷积，每个位置卷积得到的值
			// 都与该权值相关），所以对每一个权重更新值来说，需要在l.delta中找出所有与之相关的敏感度，乘以相应像素值，再求和，具体实现的方式依靠im2col_cpu()与gemm_nt()完成。
			// （backward_convolutional_layer整个函数的代码非常重要，仅靠文字没有公式与图表辅助说明可能很难说清，所以这部分更为清晰详细的说明，请参考个人博客！）
			// GEneral Matrix to Matrix Multiplication
			// 此处在im2col_cpu操作基础上，利用矩阵乘法c=alpha*a*b+beta*c完成对图像卷积的操作；
			// 0表示不对输入a进行转置，1表示对输入b进行转置；
			// m是输入a,c的行数，具体含义为卷积核的个数(l.n)；
			// n是输入b,c的列数，具体含义为每个卷积核元素个数乘以输入图像的通道数(l.size*l.size*l.c)；
			// k是输入a的列数也是b的行数，具体含义为每个输出特征图的元素个数（l.out_w*l.out_h）；
			// a,b,c即为三个参与运算的矩阵（用一维数组存储）,alpha=beta=1为常系数；
			// a为l.delta的一大行。l.delta为本层所有输出元素（包含整个batch中每张图片的所有输出特征图）关于加权输入的导数（即激活函数的导数值）集合,
			// 元素个数为l.batch * l.out_h * l.out_w * l.out_c（l.out_c = l.n），按行存储，共有l.batch行，l.out_c * l.out_h * l.out_w列，
			// 即l.delta中每行包含一张图的所有输出图，故这么一大行，又可以视作有l.out_c（l.out_c=l.n）小行，l.out_h*l*out_w小列，而一次循环就是处理l.delta的一大行，
			// 故可以将a视作l.out_c行，l.out_h*l*out_w列的矩阵；
			// b为单张输入图像经过im2col_cpu重排后的图像数据；
			// c为输出，按行存储，可视作有l.n行，l.c*l.size*l.size列（l.c是输入图像的通道数，l.n是卷积核个数），
			// 即c就是所谓的误差项（输出关于加权输入的导数），或者敏感度（强烈推荐：https://www.zybuluo.com/hanbingtao/note/485480）（一个核有l.c*l.size*l.size个权重，共有l.n个核）。
			// 由上可知：
			// a: (l.out_c) * (l.out_h*l*out_w)
			// b: (l.c * l.size * l.size) * (l.out_h * l.out_w)
			// c: (l.n) * (l.c*l.size*l.size)（注意：l.n = l.out_c）
			// 故要进行a * b + c计算，必须对b进行转置（否则行列不匹配），因故调用gemm_nt()函数
            gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);

			// 接下来，用当前层的敏感度图l.delta以及权重l.weights（还未更新）来获取上一层网络的敏感度图，BP算法的主要流程就是依靠这种层与层之间敏感度反向递推传播关系来实现。
			// 在network.c的backward_network()中，会从最后一层网络往前遍循环历至第一层，而每次开始遍历某一层网络之前，都会更新net.input为这一层网络前一层的输出，即prev.output,
			// 同时更新net.delta为prev.delta，因此，这里的net.delta是当前层前一层的敏感度图。
			// 已经强调很多次了，再说一次：下面得到的上一层的敏感度并不完整，完整的敏感度图是损失函数对上一层的加权输入的导数，
			// 而这里得到的敏感度图是损失函数对上一层输出值的导数，还差乘以一个输出值也即激活函数对加权输入的导数。
            if (state.delta) {
				// 当前层还未更新的权重
                a = l.weights + j*l.nweights / l.groups;
				
				// 每次循环仅处理一张输入图，注意移位（l.delta的维度为l.batch * l.out_c * l.out_w * l.out_h）（注意l.n = l.out_c，另外提一下，对整个网络来说，每一层的l.batch其实都是一样的）
                b = l.delta + (i*l.groups + j)*m*k;
				
				// net.workspace和上面一样，还是一张输入图片的重排，不同的是，此处我们只需要这个容器，而里面存储的值我们并不需要，在后面的处理过程中，
				// 会将其中存储的值一一覆盖掉（尺寸维持不变，还是(l.c * l.size * l.size) * (l.out_h * l.out_w）
                c = state.workspace;
				
				 // 相比上一个gemm，此处的a对应上一个的c,b对应上一个的a，c对应上一个的b，即此处a,b,c的行列分别为：
				// a: (l.n) * (l.c*l.size*l.size)，表示当前层所有权重系数
				// b: (l.out_c) * (l.out_h*l*out_w)（注意：l.n = l.out_c），表示当前层的敏感度图
				// c: (l.c * l.size * l.size) * (l.out_h * l.out_w)，表示上一层的敏感度图（其元素个数等于上一层网络单张输入图片的所有输出元素个数），
				// 此时要完成a * b + c计算，必须对a进行转置（否则行列不匹配），因故调用gemm_tn()函数。
				// 此操作含义是用：用当前层还未更新的权重值对敏感度图做卷积，得到包含上一层所有敏感度信息的矩阵，但这不是上一层最终的敏感度图，
				// 因为此时的c，也即net.workspace的尺寸为(l.c * l.size * l.size) * (l.out_h * l.out_w)，明显不是上一层的输出尺寸l.c*l.w*l.h，
				// 接下来还需要调用col2im_cpu()函数将其恢复至l.c*l.w*l.h（可视为l.c行，l.w*l.h列），这才是上一层的敏感度图（实际还差一个环节，
				// 这个环节需要等到下一次调用backward_convolutional_layer()才完成：将net.delta中每个元素乘以激活函数对加权输入的导数值）。
				// 完成gemm这一步，如col2im_cpu()中注释，是考虑了多个卷积核导致的一对多关系（上一层的一个输出元素会流入到下一层多个输出元素中），
				// 接下来调用col2im_cpu()则是考虑卷积核重叠（步长较小）导致的一对多关系。
                gemm(1, 0, n, k, m, 1, a, n, b, k, 0, c, k);

                //col2im_cpu(state.workspace, l.c / l.groups, l.h, l.w, l.size, l.stride,
                //     l.pad, state.delta + (i*l.groups + j)*l.c / l.groups*l.h*l.w);
				
				// 对c也即state.workspace进行重排，得到的结果存储在state.delta中，每次循环只会处理一张输入图片，因此，此处只会得到一张输入图产生的敏感图（注意net.delta的移位）,
				// 整个循环结束后，net.delta的总尺寸为l.batch * l.h * l.w * l.c，这就是上一层网络整个batch的敏感度图，可视为有l.batch行，l.h*l.w*l.c列，
				// 每行存储了一张输入图片所有输出特征图的敏感度
				// col2im_cpu()函数中会调用col2im_add_pixel()函数，该函数中使用了+=运算符，也即该函数要求输入的net.delta的初始值为0,而在gradient_array()中注释到l.delta的元素是不为0（也不能为0）的，
				// 看上去是矛盾的，实则不然，gradient_array()使用的l.delta是当前层的敏感度图，而在col2im_cpu()使用的net.delta是上一层的敏感度图，正如gradient_array()中所注释的，
				// 当前层l.delta之所以不为0,是因为从后面层反向传播过来的，对于上一层，显然还没有反向传播到那，因此net.delta的初始值都是为0的（注意，每一层在构建时，就为其delta动态分配了内存，
				// 且在前向传播时，为每一层的delta都赋值为0,可以参考network.c中forward_network()函数）
                col2im_cpu_ext(
                    state.workspace,        // input
                    l.c / l.groups,         // input channels (h, w)
                    l.h, l.w,               // input size (h, w)
                    l.size, l.size,         // kernel size (h, w)
                    l.pad, l.pad,           // padding (h, w)
                    l.stride_y, l.stride_x,     // stride (h, w)
                    l.dilation, l.dilation, // dilation (h, w)
                    state.delta + (i*l.groups + j)* (l.c / l.groups)*l.h*l.w); // output (delta)
            }
        }
    }
}

void update_convolutional_layer(convolutional_layer l, int batch, float learning_rate_init, float momentum, float decay)
{
    float learning_rate = learning_rate_init*l.learning_rate_scale;
    //float momentum = a.momentum;
    //float decay = a.decay;
    //int batch = a.batch;

    axpy_cpu(l.nweights, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.nweights, learning_rate / batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.nweights, momentum, l.weight_updates, 1);

    axpy_cpu(l.n, learning_rate / batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    if (l.scales) {
        axpy_cpu(l.n, learning_rate / batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }
}



image get_convolutional_weight(convolutional_layer l, int i)
{
    int h = l.size;
    int w = l.size;
    int c = l.c / l.groups;
    return float_to_image(w, h, c, l.weights + i*h*w*c);
}

void rgbgr_weights(convolutional_layer l)
{
    int i;
    for (i = 0; i < l.n; ++i) {
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            rgbgr_image(im);
        }
    }
}

void rescale_weights(convolutional_layer l, float scale, float trans)
{
    int i;
    for (i = 0; i < l.n; ++i) {
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            scale_image(im, scale);
            float sum = sum_array(im.data, im.w*im.h*im.c);
            l.biases[i] += sum*trans;
        }
    }
}

image *get_weights(convolutional_layer l)
{
    image *weights = (image *)xcalloc(l.n, sizeof(image));
    int i;
    for (i = 0; i < l.n; ++i) {
        weights[i] = copy_image(get_convolutional_weight(l, i));
        normalize_image(weights[i]);
        /*
        char buff[256];
        sprintf(buff, "filter%d", i);
        save_image(weights[i], buff);
        */
    }
    //error("hey");
    return weights;
}

image *visualize_convolutional_layer(convolutional_layer l, char *window, image *prev_weights)
{
    image *single_weights = get_weights(l);
    show_images(single_weights, l.n, window);

    image delta = get_convolutional_image(l);
    image dc = collapse_image_layers(delta, 1);
    char buff[256];
    sprintf(buff, "%s: Output", window);
    show_image(dc, buff);
    //save_image(dc, buff);
    free_image(dc);
    return single_weights;
}

