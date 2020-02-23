#include "batchnorm_layer.h"
#include "blas.h"
#include "utils.h"
#include <stdio.h>


layer make_batchnorm_layer(int batch, int w, int h, int c, int train)
{
    fprintf(stderr, "Batch Normalization Layer: %d x %d x %d image\n", w,h,c);
    layer layer = { (LAYER_TYPE)0 }; 
    layer.type = BATCHNORM; // 网络层的名字
    layer.batch = batch; //一个batch中包含的图片数
    layer.train = train; 
    layer.h = layer.out_h = h;  // 当前层的输出高度等于输入高度h
    layer.w = layer.out_w = w; // 当前层的输出宽度等于输入宽度w
    layer.c = layer.out_c = c; // 当前层的输出通道数等于输入通道数

    layer.n = layer.c;
    layer.output = (float*)xcalloc(h * w * c * batch, sizeof(float)); // layer.output为该层所有的输出（包括mini-batch所有输入图片的输出）
    layer.delta = (float*)xcalloc(h * w * c * batch, sizeof(float)); //layer.delta 是该层的敏感度图，和输出的维度想同
    layer.inputs = w*h*c; //mini-batch中每张输入图片的像素元素个数
    layer.outputs = layer.inputs; // 对应每张输入图片的所有输出特征图的总元素个数（每张输入图片会得到n也即layer.out_c张特征图）

    layer.biases = (float*)xcalloc(c, sizeof(float)); // BN层特有参数，缩放系数
    layer.bias_updates = (float*)xcalloc(c, sizeof(float)); // 缩放系数的敏感度图

    layer.scales = (float*)xcalloc(c, sizeof(float)); // BN层特有参数，偏置系数
    layer.scale_updates = (float*)xcalloc(c, sizeof(float)); // 偏置系数的敏感度图
    int i;
    for(i = 0; i < c; ++i){
        layer.scales[i] = 1; // 将缩放系数初始化为1
    }
 
    layer.mean = (float*)xcalloc(c, sizeof(float)); // mean 一个batch中所有图片的均值，分通道求取
    layer.variance = (float*)xcalloc(c, sizeof(float));  // variance 一个batch中所有图片的方差，分通道求取

    layer.rolling_mean = (float*)xcalloc(c, sizeof(float)); // 均值的滑动平均，影子变量
    layer.rolling_variance = (float*)xcalloc(c, sizeof(float)); // 方差的滑动平均，影子变量

    layer.forward = forward_batchnorm_layer; // 前向传播函数
    layer.backward = backward_batchnorm_layer; // 反向传播函数
    layer.update = update_batchnorm_layer;
#ifdef GPU
    layer.forward_gpu = forward_batchnorm_layer_gpu;
    layer.backward_gpu = backward_batchnorm_layer_gpu;
    layer.update_gpu = update_batchnorm_layer_gpu;

    layer.output_gpu =  cuda_make_array(layer.output, h * w * c * batch);

    layer.biases_gpu = cuda_make_array(layer.biases, c);
    layer.scales_gpu = cuda_make_array(layer.scales, c);

    if (train) {
        layer.delta_gpu = cuda_make_array(layer.delta, h * w * c * batch);

        layer.bias_updates_gpu = cuda_make_array(layer.bias_updates, c);
        layer.scale_updates_gpu = cuda_make_array(layer.scale_updates, c);

        layer.mean_delta_gpu = cuda_make_array(layer.mean, c);
        layer.variance_delta_gpu = cuda_make_array(layer.variance, c);
    }

    layer.mean_gpu = cuda_make_array(layer.mean, c);
    layer.variance_gpu = cuda_make_array(layer.variance, c);

    layer.rolling_mean_gpu = cuda_make_array(layer.mean, c);
    layer.rolling_variance_gpu = cuda_make_array(layer.variance, c);

    if (train) {
        layer.x_gpu = cuda_make_array(layer.output, layer.batch*layer.outputs);
#ifndef CUDNN
        layer.x_norm_gpu = cuda_make_array(layer.output, layer.batch*layer.outputs);
#endif  // not CUDNN
    }

#ifdef CUDNN
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&layer.normTensorDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&layer.normDstTensorDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(layer.normDstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, layer.batch, layer.out_c, layer.out_h, layer.out_w));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(layer.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, layer.out_c, 1, 1));
#endif
#endif
    return layer;
}

// 求gamma的梯度,对应公式 BN 2-6
//backward_scale_cpu(l.x_norm, l.delta, l.batch, l.out_c, l.out_w*l.out_h, l.scale_updates);
// x_norm 代表BN层前向传播的输出值
// delta 代表上一层的梯度图
// batch 为l.batch，即一个batch的图片数
// n代表输出通道数，也即是输入通道数
// size 代表w * h
// scale_updates 代表scale的梯度更新值
// y = gamma * x + beta
// dy / d(gamma) = x
void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
{
    int i,b,f;
    for(f = 0; f < n; ++f){
        float sum = 0;
        for(b = 0; b < batch; ++b){
            for(i = 0; i < size; ++i){
                int index = i + size*(f + n*b);
                sum += delta[index] * x_norm[index];
            }
        }
        scale_updates[f] += sum;
    }
}

// 求y对均值的导数,对应公式 BN 2-2
//不过Darknet特殊的点在于是先计算均值的梯度
// 这个时候方差是没有梯度的，所以公式3的后半部分为0，也就只保留了公式3的前半部分
// 不过我从理论上无法解释这种操作会带来什么影响，但从目标检测来看应该是没有影响的
void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{

    int i,j,k;
    for(i = 0; i < filters; ++i){
        mean_delta[i] = 0;
        for (j = 0; j < batch; ++j) {
            for (k = 0; k < spatial; ++k) {
                int index = j*filters*spatial + i*spatial + k;
                mean_delta[i] += delta[index];
            }
        }
        mean_delta[i] *= (-1./sqrt(variance[i] + .00001f));
    }
}

// 求y对方差的导数,对应公式 BN 2-1
void  variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{

    int i,j,k;
    for(i = 0; i < filters; ++i){
        variance_delta[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                variance_delta[i] += delta[index]*(x[index] - mean[i]);
            }
        }
        variance_delta[i] *= -.5 * pow(variance[i] + .00001f, (float)(-3./2.));
    }
}

// 求出BN层的梯度敏感度图
// 对应了论文中的求导公式BN 1-6，即是对x_i求导
void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
{
    int f, j, k;
    for(j = 0; j < batch; ++j){
        for(f = 0; f < filters; ++f){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + f*spatial + k;
                delta[index] = delta[index] * 1./(sqrt(variance[f]) + .00001f) + variance_delta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
            }
        }
    }
}

void resize_batchnorm_layer(layer *l, int w, int h)
{
    l->out_h = l->h = h;
    l->out_w = l->w = w;
    l->outputs = l->inputs = h*w*l->c;

    const int output_size = l->outputs * l->batch;

    l->output = (float*)realloc(l->output, output_size * sizeof(float));
    l->delta = (float*)realloc(l->delta, output_size * sizeof(float));

#ifdef GPU
    cuda_free(l->output_gpu);
    l->output_gpu = cuda_make_array(l->output, output_size);

    if (l->train) {
        cuda_free(l->delta_gpu);
        l->delta_gpu = cuda_make_array(l->delta, output_size);

        cuda_free(l->x_gpu);
        l->x_gpu = cuda_make_array(l->output, output_size);
#ifndef CUDNN
        cuda_free(l->x_norm_gpu);
        l->x_norm_gpu = cuda_make_array(l->output, output_size);
#endif  // not CUDNN
    }


#ifdef CUDNN
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(l->normDstTensorDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->normDstTensorDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->normDstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w));
#endif // CUDNN
#endif // GPU
}

// BN层的前向传播函数
void forward_batchnorm_layer(layer l, network_state state)
{
    if(l.type == BATCHNORM) copy_cpu(l.outputs*l.batch, state.input, 1, l.output, 1);
    if(l.type == CONNECTED){
        l.out_c = l.outputs;
        l.out_h = l.out_w = 1;
    }
	// 训练阶段
    if(state.train){
		// blas.c中有详细注释，计算输入数据的均值，保存为l.mean
        mean_cpu(l.output, l.batch, l.out_c, l.out_h*l.out_w, l.mean);
		// blas.c中有详细注释，计算输入数据的方差，保存为l.variance
        variance_cpu(l.output, l.mean, l.batch, l.out_c, l.out_h*l.out_w, l.variance);

		// 计算滑动平均和方差，影子变量
        scal_cpu(l.out_c, .9, l.rolling_mean, 1);
        axpy_cpu(l.out_c, .1, l.mean, 1, l.rolling_mean, 1);
        scal_cpu(l.out_c, .9, l.rolling_variance, 1);
        axpy_cpu(l.out_c, .1, l.variance, 1, l.rolling_variance, 1);

        copy_cpu(l.outputs*l.batch, l.output, 1, l.x, 1);
		// 减去均值，除以方差得到x^，论文中的第3个公式
        normalize_cpu(l.output, l.mean, l.variance, l.batch, l.out_c, l.out_h*l.out_w);
		// BN层的输出
        copy_cpu(l.outputs*l.batch, l.output, 1, l.x_norm, 1);
    } else {
		// 测试阶段，直接用滑动变量来计算输出
        normalize_cpu(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.out_c, l.out_h*l.out_w);
    }
	// 最后一个公式，对输出进行移位和偏置
    scale_bias(l.output, l.scales, l.batch, l.out_c, l.out_h*l.out_w);
    add_bias(l.output, l.biases, l.batch, l.out_c, l.out_w*l.out_h);
}

// BN层的反向传播函数
void backward_batchnorm_layer(const layer l, network_state state)
{
	// 这里是对论文中最后一个公式的缩放系数求梯度更新值
    backward_scale_cpu(l.x_norm, l.delta, l.batch, l.out_c, l.out_w*l.out_h, l.scale_updates);
	// 也是在convlution_layer.c中定义的函数，先将敏感度图乘以l.scales
    scale_bias(l.delta, l.scales, l.batch, l.out_c, l.out_h*l.out_w);
	
	//对均值求倒数
    mean_delta_cpu(l.delta, l.variance, l.batch, l.out_c, l.out_w*l.out_h, l.mean_delta);
    //对方差求倒数
	variance_delta_cpu(l.x, l.delta, l.mean, l.variance, l.batch, l.out_c, l.out_w*l.out_h, l.variance_delta);
    //计算敏感度图，对应了论文中的最后一部分
	normalize_delta_cpu(l.x, l.mean, l.variance, l.mean_delta, l.variance_delta, l.batch, l.out_c, l.out_w*l.out_h, l.delta);
    if(l.type == BATCHNORM) copy_cpu(l.outputs*l.batch, l.delta, 1, state.delta, 1);
}

void update_batchnorm_layer(layer l, int batch, float learning_rate, float momentum, float decay)
{
    //int size = l.nweights;
    axpy_cpu(l.c, learning_rate / batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.c, momentum, l.bias_updates, 1);

    axpy_cpu(l.c, learning_rate / batch, l.scale_updates, 1, l.scales, 1);
    scal_cpu(l.c, momentum, l.scale_updates, 1);
}




#ifdef GPU

void pull_batchnorm_layer(layer l)
{
    cuda_pull_array(l.biases_gpu, l.biases, l.c);
    cuda_pull_array(l.scales_gpu, l.scales, l.c);
    cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.c);
    cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.c);
}
void push_batchnorm_layer(layer l)
{
    cuda_push_array(l.biases_gpu, l.biases, l.c);
    cuda_push_array(l.scales_gpu, l.scales, l.c);
    cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.c);
    cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.c);
}

void forward_batchnorm_layer_gpu(layer l, network_state state)
{
    if (l.type == BATCHNORM) simple_copy_ongpu(l.outputs*l.batch, state.input, l.output_gpu);
        //copy_ongpu(l.outputs*l.batch, state.input, 1, l.output_gpu, 1);

    if (state.train) {
        simple_copy_ongpu(l.outputs*l.batch, l.output_gpu, l.x_gpu);
#ifdef CUDNN
        float one = 1;
        float zero = 0;
        cudnnBatchNormalizationForwardTraining(cudnn_handle(),
            CUDNN_BATCHNORM_SPATIAL,
            &one,
            &zero,
            l.normDstTensorDesc,
            l.x_gpu,                // input
            l.normDstTensorDesc,
            l.output_gpu,            // output
            l.normTensorDesc,
            l.scales_gpu,
            l.biases_gpu,
            .01,
            l.rolling_mean_gpu,        // output (should be FP32)
            l.rolling_variance_gpu,    // output (should be FP32)
            .00001,
            l.mean_gpu,            // output (should be FP32)
            l.variance_gpu);    // output (should be FP32)

        if (state.net.try_fix_nan) {
            fix_nan_and_inf(l.scales_gpu, l.n);
            fix_nan_and_inf(l.biases_gpu, l.n);
            fix_nan_and_inf(l.mean_gpu, l.n);
            fix_nan_and_inf(l.variance_gpu, l.n);
            fix_nan_and_inf(l.rolling_mean_gpu, l.n);
            fix_nan_and_inf(l.rolling_variance_gpu, l.n);
        }
#else   // CUDNN
        fast_mean_gpu(l.output_gpu, l.batch, l.out_c, l.out_h*l.out_w, l.mean_gpu);
        fast_variance_gpu(l.output_gpu, l.mean_gpu, l.batch, l.out_c, l.out_h*l.out_w, l.variance_gpu);

        scal_ongpu(l.out_c, .99, l.rolling_mean_gpu, 1);
        axpy_ongpu(l.out_c, .01, l.mean_gpu, 1, l.rolling_mean_gpu, 1);
        scal_ongpu(l.out_c, .99, l.rolling_variance_gpu, 1);
        axpy_ongpu(l.out_c, .01, l.variance_gpu, 1, l.rolling_variance_gpu, 1);

        copy_ongpu(l.outputs*l.batch, l.output_gpu, 1, l.x_gpu, 1);
        normalize_gpu(l.output_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
        copy_ongpu(l.outputs*l.batch, l.output_gpu, 1, l.x_norm_gpu, 1);

        scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
#endif  // CUDNN
    }
    else {
        normalize_gpu(l.output_gpu, l.rolling_mean_gpu, l.rolling_variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
        scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
    }

}

void backward_batchnorm_layer_gpu(layer l, network_state state)
{
    if (!state.train) {
        l.mean_gpu = l.rolling_mean_gpu;
        l.variance_gpu = l.rolling_variance_gpu;
    }
#ifdef CUDNN
    float one = 1;
    float zero = 0;
    cudnnBatchNormalizationBackward(cudnn_handle(),
        CUDNN_BATCHNORM_SPATIAL,
        &one,
        &zero,
        &one,
        &one,
        l.normDstTensorDesc,
        l.x_gpu,                // input
        l.normDstTensorDesc,
        l.delta_gpu,            // input
        l.normDstTensorDesc,
        l.output_gpu, //l.x_norm_gpu,            // output
        l.normTensorDesc,
        l.scales_gpu,            // input (should be FP32)
        l.scale_updates_gpu,    // output (should be FP32)
        l.bias_updates_gpu,        // output (should be FP32)
        .00001,
        l.mean_gpu,                // input (should be FP32)
        l.variance_gpu);        // input (should be FP32)
    simple_copy_ongpu(l.outputs*l.batch, l.output_gpu, l.delta_gpu);
    //simple_copy_ongpu(l.outputs*l.batch, l.x_norm_gpu, l.delta_gpu);
#else   // CUDNN
    backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.out_c, l.out_w*l.out_h);
    backward_scale_gpu(l.x_norm_gpu, l.delta_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.scale_updates_gpu);

    scale_bias_gpu(l.delta_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);

    fast_mean_delta_gpu(l.delta_gpu, l.variance_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.mean_delta_gpu);
    fast_variance_delta_gpu(l.x_gpu, l.delta_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.variance_delta_gpu);
    normalize_delta_gpu(l.x_gpu, l.mean_gpu, l.variance_gpu, l.mean_delta_gpu, l.variance_delta_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.delta_gpu);
#endif  // CUDNN
    if (l.type == BATCHNORM) simple_copy_ongpu(l.outputs*l.batch, l.delta_gpu, state.delta);
        //copy_ongpu(l.outputs*l.batch, l.delta_gpu, 1, state.delta, 1);

    if (state.net.try_fix_nan) {
        fix_nan_and_inf(l.scale_updates_gpu, l.n);
        fix_nan_and_inf(l.bias_updates_gpu, l.n);
    }
}

void update_batchnorm_layer_gpu(layer l, int batch, float learning_rate_init, float momentum, float decay)
{
    float learning_rate = learning_rate_init*l.learning_rate_scale;
    //float momentum = a.momentum;
    //float decay = a.decay;
    //int batch = a.batch;

    axpy_ongpu(l.c, learning_rate / batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
    scal_ongpu(l.c, momentum, l.bias_updates_gpu, 1);

    axpy_ongpu(l.c, learning_rate / batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
    scal_ongpu(l.c, momentum, l.scale_updates_gpu, 1);
}

#endif  // GPU
