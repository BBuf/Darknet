#include "blas.h"
#include "utils.h"

#include <math.h>
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void reorg_cpu(float *x, int out_w, int out_h, int out_c, int batch, int stride, int forward, float *out)
{
    int b,i,j,k;
    int in_c = out_c/(stride*stride);

    //printf("\n out_c = %d, out_w = %d, out_h = %d, stride = %d, forward = %d \n", out_c, out_w, out_h, stride, forward);
    //printf("  in_c = %d,  in_w = %d,  in_h = %d \n", in_c, out_w*stride, out_h*stride);

    for(b = 0; b < batch; ++b){
        for(k = 0; k < out_c; ++k){
            for(j = 0; j < out_h; ++j){
                for(i = 0; i < out_w; ++i){
                    int in_index  = i + out_w*(j + out_h*(k + out_c*b));
                    int c2 = k % in_c;
                    int offset = k / in_c;
                    int w2 = i*stride + offset % stride;
                    int h2 = j*stride + offset / stride;
                    int out_index = w2 + out_w*stride*(h2 + out_h*stride*(c2 + in_c*b));
                    if(forward) out[out_index] = x[in_index];    // used by default for forward (i.e. forward = 0)
                    else out[in_index] = x[out_index];
                }
            }
        }
    }
}

void flatten(float *x, int size, int layers, int batch, int forward)
{
    float* swap = (float*)xcalloc(size * layers * batch, sizeof(float));
    int i,c,b;
    for(b = 0; b < batch; ++b){
        for(c = 0; c < layers; ++c){
            for(i = 0; i < size; ++i){
                int i1 = b*layers*size + c*size + i;
                int i2 = b*layers*size + i*layers + c;
                if (forward) swap[i2] = x[i1];
                else swap[i1] = x[i2];
            }
        }
    }
    memcpy(x, swap, size*layers*batch*sizeof(float));
    free(swap);
}

void weighted_sum_cpu(float *a, float *b, float *s, int n, float *c)
{
    int i;
    for(i = 0; i < n; ++i){
        c[i] = s[i]*a[i] + (1-s[i])*(b ? b[i] : 0);
    }
}

void weighted_delta_cpu(float *a, float *b, float *s, float *da, float *db, float *ds, int n, float *dc)
{
    int i;
    for(i = 0; i < n; ++i){
        if(da) da[i] += dc[i] * s[i];
        if(db) db[i] += dc[i] * (1-s[i]);
        ds[i] += dc[i] * (a[i] - b[i]);
    }
}

static float relu(float src) {
    if (src > 0) return src;
    return 0;
}

void shortcut_multilayer_cpu(int size, int src_outputs, int batch, int n, int *outputs_of_layers, float **layers_output, float *out, float *in, float *weights, int nweights, WEIGHTS_NORMALIZATION_T weights_normalizion)
{
    // nweights - l.n or l.n*l.c or (l.n*l.c*l.h*l.w)
    const int layer_step = nweights / (n + 1);    // 1 or l.c or (l.c * l.h * l.w)
    int step = 0;
    if (nweights > 0) step = src_outputs / layer_step; // (l.c * l.h * l.w) or (l.w*l.h) or 1

    int id;
    #pragma omp parallel for
    for (id = 0; id < size; ++id) {

        int src_id = id;
        const int src_i = src_id % src_outputs;
        src_id /= src_outputs;
        int src_b = src_id;

        float sum = 1, max_val = -FLT_MAX;
        int i;
        if (weights && weights_normalizion) {
            if (weights_normalizion == SOFTMAX_NORMALIZATION) {
                for (i = 0; i < (n + 1); ++i) {
                    const int weights_index = src_i / step + i*layer_step;  // [0 or c or (c, h ,w)]
                    float w = weights[weights_index];
                    if (max_val < w) max_val = w;
                }
            }
            const float eps = 0.0001;
            sum = eps;
            for (i = 0; i < (n + 1); ++i) {
                const int weights_index = src_i / step + i*layer_step;  // [0 or c or (c, h ,w)]
                const float w = weights[weights_index];
                if (weights_normalizion == RELU_NORMALIZATION) sum += relu(w);
                else if (weights_normalizion == SOFTMAX_NORMALIZATION) sum += expf(w - max_val);
            }
        }

        if (weights) {
            float w = weights[src_i / step];
            if (weights_normalizion == RELU_NORMALIZATION) w = relu(w) / sum;
            else if (weights_normalizion == SOFTMAX_NORMALIZATION) w = expf(w - max_val) / sum;

            out[id] = in[id] * w; // [0 or c or (c, h ,w)]
        }
        else out[id] = in[id];

        // layers
        for (i = 0; i < n; ++i) {
            int add_outputs = outputs_of_layers[i];
            if (src_i < add_outputs) {
                int add_index = add_outputs*src_b + src_i;
                int out_index = id;

                float *add = layers_output[i];

                if (weights) {
                    const int weights_index = src_i / step + (i + 1)*layer_step;  // [0 or c or (c, h ,w)]
                    float w = weights[weights_index];
                    if (weights_normalizion == RELU_NORMALIZATION) w = relu(w) / sum;
                    else if (weights_normalizion == SOFTMAX_NORMALIZATION) w = expf(w - max_val) / sum;

                    out[out_index] += add[add_index] * w; // [0 or c or (c, h ,w)]
                }
                else out[out_index] += add[add_index];
            }
        }
    }
}

void backward_shortcut_multilayer_cpu(int size, int src_outputs, int batch, int n, int *outputs_of_layers,
    float **layers_delta, float *delta_out, float *delta_in, float *weights, float *weight_updates, int nweights, float *in, float **layers_output, WEIGHTS_NORMALIZATION_T weights_normalizion)
{
    // nweights - l.n or l.n*l.c or (l.n*l.c*l.h*l.w)
    const int layer_step = nweights / (n + 1);    // 1 or l.c or (l.c * l.h * l.w)
    int step = 0;
    if (nweights > 0) step = src_outputs / layer_step; // (l.c * l.h * l.w) or (l.w*l.h) or 1

    int id;
    #pragma omp parallel for
    for (id = 0; id < size; ++id) {
        int src_id = id;
        int src_i = src_id % src_outputs;
        src_id /= src_outputs;
        int src_b = src_id;

        float grad = 1, sum = 1, max_val = -FLT_MAX;;
        int i;
        if (weights && weights_normalizion) {
            if (weights_normalizion == SOFTMAX_NORMALIZATION) {
                for (i = 0; i < (n + 1); ++i) {
                    const int weights_index = src_i / step + i*layer_step;  // [0 or c or (c, h ,w)]
                    float w = weights[weights_index];
                    if (max_val < w) max_val = w;
                }
            }
            const float eps = 0.0001;
            sum = eps;
            for (i = 0; i < (n + 1); ++i) {
                const int weights_index = src_i / step + i*layer_step;  // [0 or c or (c, h ,w)]
                const float w = weights[weights_index];
                if (weights_normalizion == RELU_NORMALIZATION) sum += relu(w);
                else if (weights_normalizion == SOFTMAX_NORMALIZATION) sum += expf(w - max_val);
            }

            grad = 0;
            for (i = 0; i < (n + 1); ++i) {
                const int weights_index = src_i / step + i*layer_step;  // [0 or c or (c, h ,w)]
                const float delta_w = delta_in[id] * in[id];
                const float w = weights[weights_index];
                if (weights_normalizion == RELU_NORMALIZATION) grad += delta_w * relu(w) / sum;
                else if (weights_normalizion == SOFTMAX_NORMALIZATION) grad += delta_w * expf(w - max_val) / sum;
            }
        }

        if (weights) {
            float w = weights[src_i / step];
            if (weights_normalizion == RELU_NORMALIZATION) w = relu(w) / sum;
            else if (weights_normalizion == SOFTMAX_NORMALIZATION) w = expf(w - max_val) / sum;

            delta_out[id] += delta_in[id] * w; // [0 or c or (c, h ,w)]
            weight_updates[src_i / step] += delta_in[id] * in[id] * grad;
        }
        else delta_out[id] += delta_in[id];

        // layers
        for (i = 0; i < n; ++i) {
            int add_outputs = outputs_of_layers[i];
            if (src_i < add_outputs) {
                int add_index = add_outputs*src_b + src_i;
                int out_index = id;

                float *layer_delta = layers_delta[i];
                if (weights) {
                    float *add = layers_output[i];

                    const int weights_index = src_i / step + (i + 1)*layer_step;  // [0 or c or (c, h ,w)]
                    float w = weights[weights_index];
                    if (weights_normalizion == RELU_NORMALIZATION) w = relu(w) / sum;
                    else if (weights_normalizion == SOFTMAX_NORMALIZATION) w = expf(w - max_val) / sum;

                    layer_delta[add_index] += delta_in[id] * w; // [0 or c or (c, h ,w)]
                    weight_updates[weights_index] += delta_in[id] * add[add_index] * grad;
                }
                else layer_delta[add_index] += delta_in[id];
            }
        }
    }
}

void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out)
{
    int stride = w1/w2;
    int sample = w2/w1;
    assert(stride == h1/h2);
    assert(sample == h2/h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int i,j,k,b;
    for(b = 0; b < batch; ++b){
        for(k = 0; k < minc; ++k){
            for(j = 0; j < minh; ++j){
                for(i = 0; i < minw; ++i){
                    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
                    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
                    out[out_index] += add[add_index];
                }
            }
        }
    }
}


/*
** 计算输入数据x的平均值，输出的mean是一个矢量，比如如果x是多张三通道的图片，那么mean的维度就为通道3
** 由于每次训练输入的都是一个batch的图片，因此最终会输出batch张三通道的图片，mean中的第一个元素就是第
** 一个通道上全部batch张输出特征图所有元素的平均值，本函数的用处之一就是batch normalization的第一步了
** x: 包含所有数据，比如l.output，其包含的元素个数为l.batch*l.outputs
** batch: 一个batch中包含的图片张数，即l.batch
** filters: 该层神经网络的滤波器个数，也即该层网络输出图片的通道数（比如对卷积网络来说，就是核的个数了）
** spatial: 该层神经网络每张输出特征图的尺寸，也即等于l.out_w*l.out_h
** mean: 求得的平均值，维度为filters，也即每个滤波器对应有一个均值（每个滤波器会处理所有图片）
** x的内存排布？此处还是结合batchnorm_layer.c中的forward_batch_norm_layer()函数的调用来解释，其中x为l.output，其包含的元素个数为l
** 有l.batch行，每行有l.out_c*l.out_w*l.out_h个元素，每一行又可以分成l.out_c行，l.out_w*l.out_h列，
** 那么l.mean中的每一个元素，是某一个通道上所有batch的输出的平均值
** （比如卷积层，有3个核，那么输出通道有3个，每张输入图片都会输出3张特征图，可以理解每张输出图片是3通道的，
** 若每次输入batch=64张图片，那么将会输出64张3通道的图片，而mean中的每个元素就是某个通道上所有64张图片
** 所有元素的平均值，比如第1个通道上，所有64张图片像素平均值）
*/
void mean_cpu(float *x, int batch, int filters, int spatial, float *mean)
{
    // scale即是均值中的分母项
    float scale = 1./(batch * spatial);
    int i,j,k;
    // 外循环次数为filters，也即mean的维度，每次循环将得到一个平均值
    for(i = 0; i < filters; ++i){
        mean[i] = 0;
        // 中间循环次数为batch，也即叠加每张输入图片对应的某一通道上的输出
        for(j = 0; j < batch; ++j){
            // 内层循环即叠加一张输出特征图的所有像素值
            for(k = 0; k < spatial; ++k){
                // 计算偏移
                int index = j*filters*spatial + i*spatial + k;
                mean[i] += x[index];
            }
        }
        mean[i] *= scale;
    }
}

/*
** 计算输入x中每个元素的方差
** 本函数的主要用处应该就是batch normalization的第二步了
** x: 包含所有数据，比如l.output，其包含的元素个数为l.batch*l.outputs
** batch: 一个batch中包含的图片张数，即l.batch
** filters: 该层神经网络的滤波器个数，也即是该网络层输出图片的通道数
** spatial: 该层神经网络每张特征图的尺寸，也即等于l.out_w*l.out_h
** mean: 求得的平均值，维度为filters，也即每个滤波器对应有一个均值（每个滤波器会处理所有图片）
*/
void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    // 这里计算方差分母要减去1的原因是无偏估计，可以看：https://www.zhihu.com/question/20983193
    // 事实上，在统计学中，往往采用的方差计算公式都会让分母减1,这时因为所有数据的方差是基于均值这个固定点来计算的，
    // 对于有n个数据的样本，在均值固定的情况下，其采样自由度为n-1（只要n-1个数据固定，第n个可以由均值推出）
    float scale = 1./(batch * spatial - 1);
    int i,j,k;
    for(i = 0; i < filters; ++i){
        variance[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                // 每个元素减去均值求平方
                variance[i] += pow((x[index] - mean[i]), 2);
            }
        }
        variance[i] *= scale;
    }
}
void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    float scale = 1./(batch * spatial - 1);
    int i,j,k;
    for(i = 0; i < filters; ++i){
        variance[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                variance[i] += pow((x[index] - mean[i]), 2);
            }
        }
        variance[i] *= scale;
    }
}

void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    int b, f, i;
    for(b = 0; b < batch; ++b){
        for(f = 0; f < filters; ++f){
            for(i = 0; i < spatial; ++i){
                int index = b*filters*spatial + f*spatial + i;
                x[index] = (x[index] - mean[f])/(sqrt(variance[f]) + .000001f);
            }
        }
    }
}

void const_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void mul_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] *= X[i*INCX];
}

void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = pow(X[i*INCX], ALPHA);
}

/*
** axpy 是线性代数中的一种基本操作(仿射变换)完成y= alpha*x + y操作，其中x,y为矢量，alpha为实数系数，
** 请看: https://www.jianshu.com/p/e3f386771c51
** N: X中包含的有效元素个数
** ALPHA: 系数alpha
** X: 参与运算的矢量X
** INCX: 步长(倍数步长)，即x中凡是INCX倍数编号的参与运算
** Y: 参与运算的矢量，也相当于是输出
*/
void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] += ALPHA*X[i*INCX];
}

void scal_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] *= ALPHA;
}

void scal_add_cpu(int N, float ALPHA, float BETA, float *X, int INCX)
{
    int i;
    for (i = 0; i < N; ++i) X[i*INCX] = X[i*INCX] * ALPHA + BETA;
}

/*
** 初始化X数组所有元素的值为ALPHA
** N: X中包含的有效元素个数
** X: 待初始化的float数组指针
** INCX: 步长(倍数步长)，即X中凡是INCX的倍数编号进行初始化赋值操作
*/
void fill_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    if (INCX == 1 && ALPHA == 0) {
        memset(X, 0, N * sizeof(float));
    }
    else {
        for (i = 0; i < N; ++i) X[i*INCX] = ALPHA;
    }
}

void deinter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i, j;
    int index = 0;
    for(j = 0; j < B; ++j) {
        for(i = 0; i < NX; ++i){
            if(X) X[j*NX + i] += OUT[index];
            ++index;
        }
        for(i = 0; i < NY; ++i){
            if(Y) Y[j*NY + i] += OUT[index];
            ++index;
        }
    }
}

void inter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i, j;
    int index = 0;
    for(j = 0; j < B; ++j) {
        for(i = 0; i < NX; ++i){
            OUT[index++] = X[j*NX + i];
        }
        for(i = 0; i < NY; ++i){
            OUT[index++] = Y[j*NY + i];
        }
    }
}

/*
** 将输入X中的数据赋值到Y中(值复制，并不是指针复制，即之后X和Y之间再无关联，也即浅拷贝)
** N: X中包含的有效元素个数
** X: 待初始化的float数组指针
** INCX: 步长(倍数步长)，即X中凡是INCX的倍数编号进行初始化赋值操作
*/
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = X[i*INCX];
}

void mult_add_into_cpu(int N, float *X, float *Y, float *Z)
{
    int i;
    for(i = 0; i < N; ++i) Z[i] += X[i]*Y[i];
}

void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        float abs_val = fabs(diff);
        if(abs_val < 1) {
            error[i] = diff * diff;
            delta[i] = diff;
        }
        else {
            error[i] = 2*abs_val - 1;
            delta[i] = (diff > 0) ? 1 : -1;
        }
    }
}

void l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        error[i] = fabs(diff);
        delta[i] = diff > 0 ? 1 : -1;
    }
}

void softmax_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float t = truth[i];
        float p = pred[i];
        error[i] = (t) ? -log(p) : 0;
        delta[i] = t-p;
    }
}

void logistic_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float t = truth[i];
        float p = pred[i];
        error[i] = -t*log(p) - (1-t)*log(1-p);
        delta[i] = t-p;
    }
}

/*
** 计算预测数组与真实标签数组中每对元素的l2范数值，或者说是squared error
** 注意此函数，并没有求和，没有将所有误差加起来，而是对网络输出的每个元素计算误差的平方值
** n: 输出元素个数，也即是pred中的元素个数，也是truth中的元素个数
** pred: 网络最终的输出值，或者说网络的预测值，其中输出元素个数为n（也即最后一层网络神经元个数为n）
** truth: 真实标签值，其中元素个数为n（也即最后一层网络神经元个数为n）
** delta: 相当于本函数的输出，为网络的敏感度图（一般为cost_layer.c的敏感度图）
** error: 相当于本函数的输出，包含每个元素的squared eroor
** 这个函数一般被cost_layer.c调用，用于计算cost_layer每个输出的均方误差，除此之外，还有一个重要的操作，
** 就是计算网络最后一层的敏感度图，在darknet中，最后一层比较多的情况是cost_l
*/
void l2_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        error[i] = diff * diff;
        delta[i] = diff;
    }
}

float dot_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    float dot = 0;
    for(i = 0; i < N; ++i) dot += X[i*INCX] * Y[i*INCY];
    return dot;
}


/*
** input: 一组输入图片数据（含义见下面softmax_cpu()注释，下同）
** n: 一组输入数据含有的元素个数n = l.inputs/l.groups
** temp: 温度参数,关于softmax的温度参数，可以搜索一下softmax with temperature，应该会有很多的
** stride: 步长
** output  这一组输入图片数据对应的输出（也即l.output中与这一组输入对应的某一部分）
** 本函数实现的就是标准的softmax函数处理，唯一有点变化的就是在做指数运算之前，将每个输入元素
** 减去了该组输入元素中的最大值，以增加数值稳定性，关于此，可以参考博客：http://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability/，
** 这篇博客写的不错，博客中还提到了softmax-loss，此处没有实现（此处实现的也即博客中提到的softmax函数，将softmax-loss分开实现了）。
*/
void softmax(float *input, int n, float temp, float *output, int stride)
{
    int i;
    float sum = 0;
	// 赋初始最大值为float中的最小值-FLT_MAX
    float largest = -FLT_MAX;
    for(i = 0; i < n; ++i){
        if(input[i*stride] > largest) largest = input[i*stride];
    }
	// 上面说了为啥要减去最大值
    for(i = 0; i < n; ++i){
        float e = exp(input[i*stride]/temp - largest/temp);
        sum += e;
        output[i*stride] = e;
    }
	// 最后一步：归一化转换为概率（就是softmax函数的原型～），最后的输出结果保存在output中
    for(i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
}

/*
** 对输入Input进行softmax处理得到output
** input: softmax层所有的输入数据，即是net.input（上一层的输出）
** n: 一组数据中含有的元素个数n = l.inputs / l.groups
** batch: 一个batch中所含有的图片张数,net.batch
** batch_offset    一张输入图片含有的元素个数，即值等于l.inputs
** （所以叫做batch_offset，目的是要借助该参数在input中整张整张照片移位）
** groups: 一张输入图片的元素被分成了几组，值为l.groups（这个参数由配置文件指定，如果未指定，则默认为1）
** group_offset: 等于n/net.groups,组偏移（在每张输入图片元素中整组整组偏移）
** stride  跨度，这个参数类似于axpy_cpu()函数中的INCX参数，一定注意不同于卷积层中的l.stride，这个参数是指按照stride间隔从每组输入
** 数据中抽取元素，即会抽取所有索引为stride倍数的输入元素，而其他的输入元素，实际没有用到；stride=1时，显然，相当于没有这个参数，
** 所有输入数据都用到了（这个参数在softmax_layer层中，相当于没用，因为在forward_softmax_layer()中，调用该函数时，stride已经
** 被写死为1,并不能改，不知道还有没有其他地方使用了这个参数）
** temp: softmax的温度参数l.temperature，关于softmax的温度参数，可以搜索一下softmax with temperature，应该会有很多的
** output: 经softmax处理之后得到的输出l.output（即概率），与input具有相同的元素个数（见make_softmax_layer()），其实由此也可知，
 * stride的值必然为1,不然output的元素个数肯定少于input的元素个数（所以对于softmax来说，感觉设置stride是没有必要的，有点自相矛盾的意思）
** 这里的groups似乎是有歧义的，因为卷积里面有groups即组卷积参数，但是这里的意思是一张图片被分成了几份，似乎是AlexNet为了显存够用那个思路
** 但我去查看了AlexNet的cfg，这里groups也是设置为1的。所以个人认为直接将softmax出现的groups忘掉，然后记住只有分组卷积groups有用即可。
*/
void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    int g, b;
	// 遍历batch中的每张图片
    for(b = 0; b < batch; ++b){
        for(g = 0; g < groups; ++g){
			// 每张图片又按组遍历：一组一组遍历
            softmax(input + b*batch_offset + g*group_offset, n, temp, output + b*batch_offset + g*group_offset, stride);
        }
    }
}

void upsample_cpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
    int i, j, k, b;
    for (b = 0; b < batch; ++b) {
        for (k = 0; k < c; ++k) {
            for (j = 0; j < h*stride; ++j) {
                for (i = 0; i < w*stride; ++i) {
                    int in_index = b*w*h*c + k*w*h + (j / stride)*w + i / stride;
                    int out_index = b*w*h*c*stride*stride + k*w*h*stride*stride + j*w*stride + i;
                    if (forward) out[out_index] = scale*in[in_index];
                    else in[in_index] += scale*out[out_index];
                }
            }
        }
    }
}


void constrain_cpu(int size, float ALPHA, float *X)
{
    int i;
    for (i = 0; i < size; ++i) {
        X[i] = fminf(ALPHA, fmaxf(-ALPHA, X[i]));
    }
}

void fix_nan_and_inf_cpu(float *input, size_t size)
{
    int i;
    for (i = 0; i < size; ++i) {
        float val = input[i];
        if (isnan(val) || isinf(val))
            input[i] = 1.0f / i;  // pseudo random value
    }
}
