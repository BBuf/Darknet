#include "dropout_layer.h"
#include "utils.h"
#include "dark_cuda.h"
#include <stdlib.h>
#include <stdio.h>


/*
 * 构建dropout层
 * batch 一个batch中图片张数
 * inputs  dropout层每张输入图片的元素个数
 * probability dropout概率,即某个输入神经元被丢弃的概率,由配置文件指定;如果配置文件中未指定,则默认值为0.5(参见parse_dropout_layer()函数)
 * 返回dropout_layer
 *
 * 说明: dropout层的构建函数需要的输入参数比较少,网络输入数据尺寸h,w,c也不需要;
 * 注意: dropout层有l.inputs = l.outputs; 另外此处实现使用了inverted dropout, 不是标准的dropout
 */
dropout_layer make_dropout_layer(int batch, int inputs, float probability, int dropblock, float dropblock_size_rel, int dropblock_size_abs, int w, int h, int c)
{
    dropout_layer l = { (LAYER_TYPE)0 };
    l.type = DROPOUT;
    l.probability = probability; //丢弃概率 (1-probability 为保留概率)
    l.dropblock = dropblock; 
    l.dropblock_size_rel = dropblock_size_rel;
    l.dropblock_size_abs = dropblock_size_abs;
    if (l.dropblock) {
        l.out_w = l.w = w;
        l.out_h = l.h = h;
        l.out_c = l.c = c;

        if (l.w <= 0 || l.h <= 0 || l.c <= 0) {
            printf(" Error: DropBlock - there must be positive values for: l.w=%d, l.h=%d, l.c=%d \n", l.w, l.h, l.c);
            exit(0);
        }
    }
    l.inputs = inputs; // dropout层不会改变输入输出的个数,因此有 l.inputs == l.outputs
    l.outputs = inputs; // 虽然dropout会丢弃一些输入神经元, 但这丢弃只是置该输入元素值为0, 并没有删除
    l.batch = batch; // 一个batch中图片数量
    l.rand = (float*)xcalloc(inputs * batch, sizeof(float)); //动态分配内存,
    l.scale = 1./(1.0 - probability); //使用inverted dropout, scale取保留概率的倒数
    l.forward = forward_dropout_layer; //前向传播
    l.backward = backward_dropout_layer; // 反向传播
    #ifdef GPU
    l.forward_gpu = forward_dropout_layer_gpu;
    l.backward_gpu = backward_dropout_layer_gpu;
    l.rand_gpu = cuda_make_array(l.rand, inputs*batch);
    #endif
    if (l.dropblock) {
        if(l.dropblock_size_abs) fprintf(stderr, "dropblock       p = %.2f   l.dropblock_size_abs = %d         %4d  ->   %4d\n", probability, l.dropblock_size_abs, inputs, inputs);
        else fprintf(stderr, "dropblock       p = %.2f   l.dropblock_size_rel = %.2f         %4d  ->   %4d\n", probability, l.dropblock_size_rel, inputs, inputs);
    }
    else fprintf(stderr, "dropout       p = %.2f                  %4d  ->   %4d\n", probability, inputs, inputs);
    return l;
}

void resize_dropout_layer(dropout_layer *l, int inputs)
{
    l->inputs = l->outputs = inputs;
    l->rand = (float*)xrealloc(l->rand, l->inputs * l->batch * sizeof(float));
    #ifdef GPU
    cuda_free(l->rand_gpu);

    l->rand_gpu = cuda_make_array(l->rand, l->inputs*l->batch);
    #endif
}

/*
 * dropout层前向传播函数
 * @l 当前dropout层函数
 * @state 整个网络
 *
 * 说明:dropout层没有训练参数,因此前向传播函数比较简单,只需要完成一件事: 按指定概率 l.probability
 * 丢弃输入元素,并将保留下来的输入元素乘以比例因子scale(采用inverted dropout, 这种凡是实现更为方便,
 * 且代码接口比较统一;如果采用标准的dropout, 则测试阶段需要进入 forward_dropout_layer(),
 * 使每个输入乘以保留概率,而使用inverted dropout, 测试阶段就不需要进入到forward_dropout_layer())
 *
 * 说明: dropout层有l.inputs = l.outputs;
 */

void forward_dropout_layer(dropout_layer l, network_state state)
{
    int i;
	// 因为使用inverted dropout,所以测试阶段不需要进入forward_dropout_layer()
    if (!state.train) return;
    for(i = 0; i < l.batch * l.inputs; ++i){
		// 采样一个0-1之间均匀分布的随机数
        float r = rand_uniform(0, 1);
        l.rand[i] = r; // 每一个随机数都要保存到l.rand,之后反向传播时候会用到
        if(r < l.probability) state.input[i] = 0; // 舍弃该元素,将其值置为0, 所以这里元素的总个数并没有发生变化;
        else state.input[i] *= l.scale; //保留该输入元素,并乘以比例因子scale
    }
}

/*
 * dropout层反向传播函数
 * l 当前dropout层网络
 * state 整个网络
 *
 * 说明: dropout层的反向传播相对简单,因为其本身没有训练参数,也没有激活函数,或者说激活函数为f(x) =x,
 * 也就是激活函数关于加权输入的导数值为1, 因此其自身的误差项值以后由下一层网络反向传播时计算完了,
 * 没有必要再曾以激活函数关于加权输入的导数了.剩下要做的就是计算上一层的误差项net.delta, 这个计算也很简单;
 */
void backward_dropout_layer(dropout_layer l, network_state state)
{
    int i;
	// 如果state.delta为空,则返回(state.delta为空则说明已经反向传播到第一层了,此处所指定的第一层,是state.layers[0]
    // 也就是与输入层直接相连的第一层隐含层, 详细见 network.c 中的 forward_network()函数)
    if(!state.delta) return;
    for(i = 0; i < l.batch * l.inputs; ++i){
		// 与前向过程 forward_dropout_layer 照应,根据l.rand指示,
        float r = l.rand[i];
		// 如果r <　probability,说明舍丢弃的输入，其误差项值为0
        if(r < l.probability) state.delta[i] = 0;
		// 保留下的输入元素,其误差项值为当前层对应输出的误差项值乘以l.scale
        else state.delta[i] *= l.scale;
    }
}
