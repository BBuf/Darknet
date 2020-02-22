#include "activations.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

char *get_activation_string(ACTIVATION a)
{
    switch(a){
        case LOGISTIC:
            return "logistic";
        case LOGGY:
            return "loggy";
        case RELU:
            return "relu";
        case ELU:
            return "elu";
        case SELU:
            return "selu";
        case RELIE:
            return "relie";
        case RAMP:
            return "ramp";
        case LINEAR:
            return "linear";
        case TANH:
            return "tanh";
        case PLSE:
            return "plse";
        case LEAKY:
            return "leaky";
        case STAIR:
            return "stair";
        case HARDTAN:
            return "hardtan";
        case LHTAN:
            return "lhtan";
        default:
            break;
    }
    return "relu";
}

ACTIVATION get_activation(char *s)
{
    if (strcmp(s, "logistic")==0) return LOGISTIC;
    if (strcmp(s, "swish") == 0) return SWISH;
    if (strcmp(s, "mish") == 0) return MISH;
    if (strcmp(s, "normalize_channels") == 0) return NORM_CHAN;
    if (strcmp(s, "normalize_channels_softmax") == 0) return NORM_CHAN_SOFTMAX;
    if (strcmp(s, "normalize_channels_softmax_maxval") == 0) return NORM_CHAN_SOFTMAX_MAXVAL;
    if (strcmp(s, "loggy")==0) return LOGGY;
    if (strcmp(s, "relu")==0) return RELU;
    if (strcmp(s, "elu")==0) return ELU;
    if (strcmp(s, "selu") == 0) return SELU;
    if (strcmp(s, "relie")==0) return RELIE;
    if (strcmp(s, "plse")==0) return PLSE;
    if (strcmp(s, "hardtan")==0) return HARDTAN;
    if (strcmp(s, "lhtan")==0) return LHTAN;
    if (strcmp(s, "linear")==0) return LINEAR;
    if (strcmp(s, "ramp")==0) return RAMP;
    if (strcmp(s, "leaky")==0) return LEAKY;
    if (strcmp(s, "tanh")==0) return TANH;
    if (strcmp(s, "stair")==0) return STAIR;
    fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    return RELU;
}

float activate(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_activate(x);
        case LOGISTIC:
            return logistic_activate(x);
        case LOGGY:
            return loggy_activate(x);
        case RELU:
            return relu_activate(x);
        case ELU:
            return elu_activate(x);
        case SELU:
            return selu_activate(x);
        case RELIE:
            return relie_activate(x);
        case RAMP:
            return ramp_activate(x);
        case LEAKY:
            return leaky_activate(x);
        case TANH:
            return tanh_activate(x);
        case PLSE:
            return plse_activate(x);
        case STAIR:
            return stair_activate(x);
        case HARDTAN:
            return hardtan_activate(x);
        case LHTAN:
            return lhtan_activate(x);
    }
    return 0;
}

void activate_array(float *x, const int n, const ACTIVATION a)
{
    int i;
    if (a == LINEAR) {}
    else if (a == LEAKY) {
        #pragma omp parallel for
        for (i = 0; i < n; ++i) {
            x[i] = leaky_activate(x[i]);
        }
    }
    else if (a == LOGISTIC) {
        #pragma omp parallel for
        for (i = 0; i < n; ++i) {
            x[i] = logistic_activate(x[i]);
        }
    }
    else {
        for (i = 0; i < n; ++i) {
            x[i] = activate(x[i], a);
        }
    }
}

void activate_array_swish(float *x, const int n, float * output_sigmoid, float * output)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; ++i) {
        float x_val = x[i];
        float sigmoid = logistic_activate(x_val);
        output_sigmoid[i] = sigmoid;
        output[i] = x_val * sigmoid;
    }
}

// https://github.com/digantamisra98/Mish
void activate_array_mish(float *x, const int n, float * activation_input, float * output)
{
    const float MISH_THRESHOLD = 20;
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; ++i) {
        float x_val = x[i];
        activation_input[i] = x_val;    // store value before activation
        output[i] = x_val * tanh_activate( softplus_activate(x_val, MISH_THRESHOLD) );
    }
}

void activate_array_normalize_channels(float *x, const int n, int batch, int channels, int wh_step, float *output)
{
    int size = n / channels;

    int i;
    #pragma omp parallel for
    for (i = 0; i < size; ++i) {
        int wh_i = i % wh_step;
        int b = i / wh_step;

        const float eps = 0.0001;
        if (i < size) {
            float sum = eps;
            int k;
            for (k = 0; k < channels; ++k) {
                float val = x[wh_i + k * wh_step + b*wh_step*channels];
                if (val > 0) sum += val;
            }
            for (k = 0; k < channels; ++k) {
                float val = x[wh_i + k * wh_step + b*wh_step*channels];
                if (val > 0) val = val / sum;
                else val = 0;
                output[wh_i + k * wh_step + b*wh_step*channels] = val;
            }
        }
    }
}

void activate_array_normalize_channels_softmax(float *x, const int n, int batch, int channels, int wh_step, float *output, int use_max_val)
{
    int size = n / channels;

    int i;
    #pragma omp parallel for
    for (i = 0; i < size; ++i) {
        int wh_i = i % wh_step;
        int b = i / wh_step;

        const float eps = 0.0001;
        if (i < size) {
            float sum = eps;
            float max_val = -FLT_MAX;
            int k;
            if (use_max_val) {
                for (k = 0; k < channels; ++k) {
                    float val = x[wh_i + k * wh_step + b*wh_step*channels];
                    if (val > max_val) max_val = val;
                }
            }
            else
                max_val = 0;

            for (k = 0; k < channels; ++k) {
                float val = x[wh_i + k * wh_step + b*wh_step*channels];
                sum += expf(val - max_val);
            }
            for (k = 0; k < channels; ++k) {
                float val = x[wh_i + k * wh_step + b*wh_step*channels];
                val = expf(val - max_val) / sum;
                output[wh_i + k * wh_step + b*wh_step*channels] = val;
            }
        }
    }
}

void gradient_array_normalize_channels_softmax(float *x, const int n, int batch, int channels, int wh_step, float *delta)
{
    int size = n / channels;

    int i;
    #pragma omp parallel for
    for (i = 0; i < size; ++i) {
        int wh_i = i % wh_step;
        int b = i / wh_step;

        if (i < size) {
            float grad = 0;
            int k;
            for (k = 0; k < channels; ++k) {
                const int index = wh_i + k * wh_step + b*wh_step*channels;
                float out = x[index];
                float d = delta[index];
                grad += out*d;
            }
            for (k = 0; k < channels; ++k) {
                const int index = wh_i + k * wh_step + b*wh_step*channels;
                float d = delta[index];
                d = d * grad;
                delta[index] = d;
            }
        }
    }
}

void gradient_array_normalize_channels(float *x, const int n, int batch, int channels, int wh_step, float *delta)
{
    int size = n / channels;

    int i;
    #pragma omp parallel for
    for (i = 0; i < size; ++i) {
        int wh_i = i % wh_step;
        int b = i / wh_step;

        if (i < size) {
            float grad = 0;
            int k;
            for (k = 0; k < channels; ++k) {
                const int index = wh_i + k * wh_step + b*wh_step*channels;
                float out = x[index];
                float d = delta[index];
                grad += out*d;
            }
            for (k = 0; k < channels; ++k) {
                const int index = wh_i + k * wh_step + b*wh_step*channels;
                if (x[index] > 0) {
                    float d = delta[index];
                    d = d * grad;
                    delta[index] = d;
                }
            }
        }
    }
}

/*
** 根据不同的激活函数求取对输入的梯度（导数）
** 输入： x    激活函数接收的输入值
**       a    激活函数类型，包括的激活函数类型见activations.h中枚举类型ACTIVATION的定义
** 输出： 激活函数关于输入x的导数值
*/
float gradient(float x, ACTIVATION a)
{
	// 以下分别求取各种激活函数对输入的导数值，详见各个导数求取函数的内部注释
    switch(a){
        case LINEAR:
            return linear_gradient(x);
        case LOGISTIC:
            return logistic_gradient(x);
        case LOGGY:
            return loggy_gradient(x);
        case RELU:
            return relu_gradient(x);
        case NORM_CHAN:
            //return relu_gradient(x);
        case NORM_CHAN_SOFTMAX_MAXVAL:
            //...
        case NORM_CHAN_SOFTMAX:
            printf(" Error: should be used custom NORM_CHAN or NORM_CHAN_SOFTMAX-function for gradient \n");
            exit(0);
            return 0;
        case ELU:
            return elu_gradient(x);
        case SELU:
            return selu_gradient(x);
        case RELIE:
            return relie_gradient(x);
        case RAMP:
            return ramp_gradient(x);
        case LEAKY:
            return leaky_gradient(x);
        case TANH:
            return tanh_gradient(x);
        case PLSE:
            return plse_gradient(x);
        case STAIR:
            return stair_gradient(x);
        case HARDTAN:
            return hardtan_gradient(x);
        case LHTAN:
            return lhtan_gradient(x);
    }
    return 0;
}

/*
** 计算激活函数对加权输入的导数，并乘以delta，得到当前层最终的delta（敏感度图）
** 输入： x    当前层的所有输出（维度为l.batch * l.out_c * l.out_w * l.out_h）
**       n    l.output的维度，即为l.batch * l.out_c * l.out_w * l.out_h（包含整个batch的）
**       ACTIVATION    激活函数类型
**       delta     当前层敏感度图（与当前成输出x维度一样）
** 说明1： 该函数不但计算了激活函数对于加权输入的导数，还将该导数乘以了之前完成大部分计算的敏感度图delta（对应元素相乘），因此调用改函数之后，将得到该层最终的敏感度图
** 说明2： 这里直接利用输出值求激活函数关于输入的导数值是因为神经网络中所使用的绝大部分激活函数，其关于输入的导数值都可以描述为输出值的函数表达式，
          比如对于Sigmoid激活函数（记作f(x)），其导数值为f(x)'=f(x)*(1-f(x)),因此如果给出y=f(x)，那么f(x)'=y*(1-y)，只需要输出值y就可以了，不需要输入x的值，
          （暂时不确定darknet中有没有使用特殊的激活函数，以致于必须要输入值才能够求出导数值，在activiation.c文件中，有几个激活函数暂时没看懂，也没在网上查到）。
** 说明3： 关于l.delta的初值，可能你有注意到在看某一类型网络层的时候，比如卷积层中的backward_convolutional_layer()函数，没有发现在此之前对l.delta赋初值的语句，
**        只是用calloc为其动态分配了内存，这样的l.delta其所有元素的值都为0,那么这里使用*=运算符得到的值将恒为0。是的，如果只看某一层，或者说某一类型的层，的确有这个疑惑，
**        但是整个网络是有很多层的，且有多种类型，一般来说，不会以卷积层为最后一层，而回以COST或者REGION为最后一层，这些层中，会对l.delta赋初值，又由于l.delta是由后
**        网前逐层传播的，因此，当反向运行到某一层时，l.delta的值将都不会为0.
*/
void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta)
{
    int i;
    #pragma omp parallel for
    for(i = 0; i < n; ++i){
        delta[i] *= gradient(x[i], a);
    }
}

// https://github.com/BVLC/caffe/blob/04ab089db018a292ae48d51732dd6c66766b36b6/src/caffe/layers/swish_layer.cpp#L54-L56
void gradient_array_swish(const float *x, const int n, const float * sigmoid, float * delta)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; ++i) {
        float swish = x[i];
        delta[i] *= swish + sigmoid[i]*(1 - swish);
    }
}

// https://github.com/digantamisra98/Mish
void gradient_array_mish(const int n, const float * activation_input, float * delta)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; ++i) {
        const float MISH_THRESHOLD = 20.0f;

        // implementation from TensorFlow: https://github.com/tensorflow/addons/commit/093cdfa85d334cbe19a37624c33198f3140109ed
        // implementation from Pytorch: https://github.com/thomasbrandon/mish-cuda/blob/master/csrc/mish.h#L26-L31
        float inp = activation_input[i];
        const float sp = softplus_activate(inp, MISH_THRESHOLD);
        const float grad_sp = 1 - exp(-sp);
        const float tsp = tanh(sp);
        const float grad_tsp = (1 - tsp*tsp) * grad_sp;
        const float grad = inp * grad_tsp + tsp;
        delta[i] *= grad;


        //float x = activation_input[i];
        //float d = 2 * expf(x) + expf(2 * x) + 2;
        //float w = 4 * (x + 1) + 4 * expf(2 * x) + expf(3 * x) + expf(x)*(4 * x + 6);
        //float derivative = expf(x) * w / (d * d);
        //delta[i] *= derivative;
    }
}
