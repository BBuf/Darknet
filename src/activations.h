#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
#include "darknet.h"
#include "dark_cuda.h"
#include "math.h"

//typedef enum{
//    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN, SELU, SWISH, MISH
//}ACTIVATION;

#ifdef __cplusplus
extern "C" {
#endif
// 获得激活函数对应的字符串描述
ACTIVATION get_activation(char *s);

char *get_activation_string(ACTIVATION a);
//根据不同的激活函数类型，调用不同的激活函数处理输入元素
float activate(float x, ACTIVATION a);
//根据不同的激活函数求取对输入的梯度
float gradient(float x, ACTIVATION a);
// 计算激活函数对加权输入的导数, 并乘以delta，得到当前层最终的delta(误差项)
void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta);
void gradient_array_swish(const float *x, const int n, const float * sigmoid, float * delta);
void gradient_array_mish(const int n, const float * activation_input, float * delta);
// 用激活函数处理输入x中的每一个元素
void activate_array(float *x, const int n, const ACTIVATION a);
void activate_array_swish(float *x, const int n, float * output_sigmoid, float * output);
void activate_array_mish(float *x, const int n, float * activation_input, float * output);
void activate_array_normalize_channels(float *x, const int n, int batch, int channels, int wh_step, float *output);
void gradient_array_normalize_channels(float *x, const int n, int batch, int channels, int wh_step, float *delta);
void activate_array_normalize_channels_softmax(float *x, const int n, int batch, int channels, int wh_step, float *output, int use_max_val);
void gradient_array_normalize_channels_softmax(float *x, const int n, int batch, int channels, int wh_step, float *delta);
#ifdef GPU
void activate_array_ongpu(float *x, int n, ACTIVATION a);
void activate_array_swish_ongpu(float *x, int n, float *output_sigmoid_gpu, float *output_gpu);
void activate_array_mish_ongpu(float *x, int n, float *activation_input_gpu, float *output_gpu);
void gradient_array_ongpu(float *x, int n, ACTIVATION a, float *delta);
void gradient_array_swish_ongpu(float *x, int n, float *sigmoid_gpu, float *delta);
void gradient_array_mish_ongpu(int n, float *activation_input_gpu, float *delta);
void activate_array_normalize_channels_ongpu(float *x, int n, int batch, int channels, int wh_step, float *output_gpu);
void gradient_array_normalize_channels_ongpu(float *output_gpu, int n, int batch, int channels, int wh_step, float *delta_gpu);
void activate_array_normalize_channels_softmax_ongpu(float *x, int n, int batch, int channels, int wh_step, float *output_gpu, int use_max_val);
void gradient_array_normalize_channels_softmax_ongpu(float *output_gpu, int n, int batch, int channels, int wh_step, float *delta_gpu);

#endif

/*
 * 内联函数可以加快调用的速度，但是调用多次的话，会使执行文件变大，这样会降低速度。
 * static 修饰的内联函数，一般情况下不会产生函数本身的代码，而是全部嵌入在被调用的地方。
 * 如果不加 static，则表示该函数有可能被其他编译单元所调用，所以一定会产生函数本身的代码；
 *
 * gcc的 static inline相对于static函数来说只是在调用时建议编译器进行内联展开；
 * gcc不会特意为 static inline 函数生成独立的汇编码，除非出现了必须生成不可的情况（如通过函数指针和递归调用）；
 * gcc 的static inline 函数仅能作用于文件范围内。
 */
 
static inline float stair_activate(float x)
{
    int n = floorf(x);
    if (n%2 == 0) return floorf(x/2.f);
    else return (x - n) + floorf(x/2.f);
}
static inline float hardtan_activate(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}
static inline float linear_activate(float x){return x;}
static inline float logistic_activate(float x){return 1.f/(1.f + expf(-x));}
static inline float loggy_activate(float x){return 2.f/(1.f + expf(-x)) - 1;}
static inline float relu_activate(float x){return x*(x>0);}
static inline float elu_activate(float x){return (x >= 0)*x + (x < 0)*(expf(x)-1);}
static inline float selu_activate(float x) { return (x >= 0)*1.0507f*x + (x < 0)*1.0507f*1.6732f*(expf(x) - 1); }
static inline float relie_activate(float x){return (x>0) ? x : .01f*x;}
static inline float ramp_activate(float x){return x*(x>0)+.1f*x;}
static inline float leaky_activate(float x){return (x>0) ? x : .1f*x;}
static inline float tanh_activate(float x){return (expf(2*x)-1)/(expf(2*x)+1);}
static inline float softplus_activate(float x, float threshold) {
    if (x > threshold) return x;                // too large
    else if (x < -threshold) return expf(x);    // too small
    return logf(expf(x) + 1);
}
static inline float plse_activate(float x)
{
    if(x < -4) return .01f * (x + 4);
    if(x > 4)  return .01f * (x - 4) + 1;
    return .125f*x + .5f;
}

static inline float lhtan_activate(float x)
{
    if(x < 0) return .001f*x;
    if(x > 1) return .001f*(x-1) + 1;
    return x;
}
static inline float lhtan_gradient(float x)
{
    if(x > 0 && x < 1) return 1;
    return .001f;
}

static inline float hardtan_gradient(float x)
{
    if (x > -1 && x < 1) return 1;
    return 0;
}
static inline float linear_gradient(float x){return 1;}
static inline float logistic_gradient(float x){return (1-x)*x;}
static inline float loggy_gradient(float x)
{
    float y = (x+1.f)/2.f;
    return 2*(1-y)*y;
}
static inline float stair_gradient(float x)
{
    if (floor(x) == x) return 0;
    return 1.0f;
}
static inline float relu_gradient(float x){return (x>0);}
static inline float elu_gradient(float x){return (x >= 0) + (x < 0)*(x + 1);}
static inline float selu_gradient(float x) { return (x >= 0)*1.0507f + (x < 0)*(x + 1.0507f*1.6732f); }
static inline float relie_gradient(float x){return (x>0) ? 1 : .01f;}
static inline float ramp_gradient(float x){return (x>0)+.1f;}
static inline float leaky_gradient(float x){return (x>0) ? 1 : .1f;}
static inline float tanh_gradient(float x){return 1-x*x;}
static inline float plse_gradient(float x){return (x < 0 || x > 1) ? .01f : .125f;}

#ifdef __cplusplus
}
#endif

#endif
