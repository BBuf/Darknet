#include "yolo_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "dark_cuda.h"
#include "utils.h"

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

// 构造YOLOV3的yolo层
// batch 一个batch中包含图片的张数
// w 输入图片的宽度
// h 输入图片的高度
// n 一个cell预测多少个bbox
// total total Anchor bbox的数目
// mask 使用的是0,1,2 还是
// classes 网络需要识别的物体类别数
layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int max_boxes)
{
    int i;
    layer l = { (LAYER_TYPE)0 };
    l.type = YOLO; //层类别

    l.n = n; //一个cell预测多少个bbox
    l.total = total; //anchors的数目，为9
    l.batch = batch;// 一个batch包含图片的张数
    l.h = h; // 输入图片的宽度
    l.w = w; // 输入图片的高度
    l.c = n*(classes + 4 + 1); // 输入图片的通道数, 3*(20 + 5)
    l.out_w = l.w;// 输出图片的宽度
    l.out_h = l.h;// 输出图片的高度
    l.out_c = l.c;// 输出图片的通道数
    l.classes = classes;//目标类别数
    l.cost = (float*)xcalloc(1, sizeof(float)); //yolo层总的损失
    l.biases = (float*)xcalloc(total * 2, sizeof(float)); //存储bbox的Anchor box的[w,h]
    if(mask) l.mask = mask; //yolov3有mask传入
    else{
        l.mask = (int*)xcalloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            l.mask[i] = i;
        }
    }
	//存储bbox的Anchor box的[w,h]的更新值
    l.bias_updates = (float*)xcalloc(n * 2, sizeof(float));
	// 一张训练图片经过yolo层后得到的输出元素个数（等于网格数*每个网格预测的矩形框数*每个矩形框的参数个数）
    l.outputs = h*w*n*(classes + 4 + 1);
	//一张训练图片输入到yolo层的元素个数（注意是一张图片，对于yolo_layer，输入和输出的元素个数相等）
    l.inputs = l.outputs; 
	//每张图片含有的真实矩形框参数的个数（max_boxes表示一张图片中最多有max_boxes个ground truth矩形框，每个真实矩形框有
    //5个参数，包括x,y,w,h四个定位参数，以及物体类别）,注意max_boxes是darknet程序内写死的，实际上每张图片可能
    //并没有max_boxes个真实矩形框，也能没有这么多参数，但为了保持一致性，还是会留着这么大的存储空间，只是其中的
    //值为空而已.
    l.max_boxes = max_boxes;
	// GT: max_boxes*(4+1) 存储max_boxes个bbox的信息，这里是假设图片中GT bbox的数量是
	//小于max_boxes的，这里是写死的；此处与yolov1是不同的
    l.truths = l.max_boxes*(4 + 1);    // 90*(4 + 1);
	// yolo层误差项(包含整个batch的)
    l.delta = (float*)xcalloc(batch * l.outputs, sizeof(float));
	//yolo层所有输出（包含整个batch的）
    //yolo的输出维度是l.out_w*l.out_h，等于输出的维度，输出的通道数为l.out_c，也即是输入的通道数，具体为：n*(classes+coords+1)
	//YOLO检测模型将图片分成S*S个网格，每个网格又预测B个矩形框，最后输出的就是这些网格中包含的所有矩形框的信息
    l.output = (float*)xcalloc(batch * l.outputs, sizeof(float));
	// 存储bbox的Anchor box的[w,h]的初始化,在src/parse.c中parse_yolo函数会加载cfg中Anchor尺寸
    for(i = 0; i < total*2; ++i){
        l.biases[i] = .5;
    }
	// yolo层的前向传播
    l.forward = forward_yolo_layer;
	// yolo层的反向传播
    l.backward = backward_yolo_layer;
#ifdef GPU
    l.forward_gpu = forward_yolo_layer_gpu;
    l.backward_gpu = backward_yolo_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);

    free(l.output);
    if (cudaSuccess == cudaHostAlloc(&l.output, batch*l.outputs*sizeof(float), cudaHostRegisterMapped)) l.output_pinned = 1;
    else {
        cudaGetLastError(); // reset CUDA-error
        l.output = (float*)xcalloc(batch * l.outputs, sizeof(float));
    }

    free(l.delta);
    if (cudaSuccess == cudaHostAlloc(&l.delta, batch*l.outputs*sizeof(float), cudaHostRegisterMapped)) l.delta_pinned = 1;
    else {
        cudaGetLastError(); // reset CUDA-error
        l.delta = (float*)xcalloc(batch * l.outputs, sizeof(float));
    }
#endif

    fprintf(stderr, "yolo\n");
    srand(time(0));

    return l;
}

void resize_yolo_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + 4 + 1);
    l->inputs = l->outputs;

    if (!l->output_pinned) l->output = (float*)xrealloc(l->output, l->batch*l->outputs * sizeof(float));
    if (!l->delta_pinned) l->delta = (float*)xrealloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
    if (l->output_pinned) {
        CHECK_CUDA(cudaFreeHost(l->output));
        if (cudaSuccess != cudaHostAlloc(&l->output, l->batch*l->outputs * sizeof(float), cudaHostRegisterMapped)) {
            cudaGetLastError(); // reset CUDA-error
            l->output = (float*)xcalloc(l->batch * l->outputs, sizeof(float));
            l->output_pinned = 0;
        }
    }

    if (l->delta_pinned) {
        CHECK_CUDA(cudaFreeHost(l->delta));
        if (cudaSuccess != cudaHostAlloc(&l->delta, l->batch*l->outputs * sizeof(float), cudaHostRegisterMapped)) {
            cudaGetLastError(); // reset CUDA-error
            l->delta = (float*)xcalloc(l->batch * l->outputs, sizeof(float));
            l->delta_pinned = 0;
        }
    }

    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

//获取某个矩形框的4个定位信息，即根据输入的矩形框索引从l.output中获取该矩形框的定位信息x,y,w,h
//x  yolo_layer的输出，即l.output，包含所有batch预测得到的矩形框信息
//biases 表示Anchor框的长和宽
//index 矩形框的首地址（索引，矩形框中存储的首个参数x在l.output中的索引）
//i 第几行（yolo_layer维度为l.out_w*l.out_c）
//j 第几列
//lw 特征图的宽度
//lh 特征图的高度
//w 输入图像的宽度
//h 输入图像的高度
//stride 不同的特征图具有不同的步长(即是两个grid cell之间跨的像素个数不同)

//biases中存储的是预定以的anchor box的宽和高（输入图尺度），(lw,lh)是yolo层输入的特征图尺度，
//(w,h)是整个网络输入图尺度，get_yolo_box()函数利用了论文截图中的公式，而且把结果分别利用特征
//图宽高和输入图宽高做了归一化。既然这个机制是用来限制回归，避免预测很远的目标，那么这个预测
//范围是多大呢？(b.x,by)最小是(i,j),最大是(i+1,x+1)，即中心点在特征图上最多一定一个像素（假设
//输入图下采样n得到特征图，特征图中一个像素对应输入图的n个像素）(b.w,b.h)最大是(2.7 * anchor.w,
//2.7 * anchor.h),最小就是(anchor.w,anchor.h)，这是在输入图尺寸下的值。

box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    box b;
    // ln - natural logarithm (base = e)
    // x` = t.x * lw - i;   // x = ln(x`/(1-x`))   // x - output of previous conv-layer
    // y` = t.y * lh - i;   // y = ln(y`/(1-y`))   // y - output of previous conv-layer
                            // w = ln(t.w * net.w / anchors_w); // w - output of previous conv-layer
                            // h = ln(t.h * net.h / anchors_h); // h - output of previous conv-layer
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 1*stride]) / lh;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}
//修复nan的问题
static inline float fix_nan_inf(float val)
{
    if (isnan(val) || isinf(val)) val = 0;
    return val;
}
//把val限制在[-max_val,max_val]区间中
static inline float clip_value(float val, const float max_val)
{
    if (val > max_val) val = max_val;
    else if (val < -max_val) val = -max_val;
    return val;
}
//调用方式：delta_yolo_box(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2 - truth.w*truth.h), l.w*l.h, l.iou_normalizer * class_multiplier, l.iou_loss, 1, l.max_delta);
// 计算预测边界框的误差
ious delta_yolo_box(box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride, float iou_normalizer, IOU_LOSS iou_loss, int accumulate, int max_delta)
{
    ious all_ious = { 0 };
    // i - step in layer width
    // j - step in layer height
    //  Returns a box in absolute coordinates
	// 获得第j*w+i个cell的第n个bbox在当前特征图的[x,y,w,h]
    box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
	//iou
    all_ious.iou = box_iou(pred, truth);
	//giou
    all_ious.giou = box_giou(pred, truth);
	//diou 
    all_ious.diou = box_diou(pred, truth);
	//ciou
    all_ious.ciou = box_ciou(pred, truth);
    // avoid nan in dx_box_iou
	
    if (pred.w == 0) { pred.w = 1.0; }
    if (pred.h == 0) { pred.h = 1.0; }
    if (iou_loss == MSE)    // old loss
    {
		// 计算GT bbox的tx, ty, tw, th
        float tx = (truth.x*lw - i); //和预测值匹配
        float ty = (truth.y*lh - j);
        float tw = log(truth.w*w / biases[2 * n]); //log 使大框和小框的误差影响接近
        float th = log(truth.h*h / biases[2 * n + 1]);

        // accumulate delta
		//计算tx, ty, tw, th的梯度
        delta[index + 0 * stride] += scale * (tx - x[index + 0 * stride]) * iou_normalizer;  //计算误差 delta，乘了权重系数 scale=(2-truth.w*truth.h)
        delta[index + 1 * stride] += scale * (ty - x[index + 1 * stride]) * iou_normalizer;
        delta[index + 2 * stride] += scale * (tw - x[index + 2 * stride]) * iou_normalizer;
        delta[index + 3 * stride] += scale * (th - x[index + 3 * stride]) * iou_normalizer;
    }
    else {
        // https://github.com/generalized-iou/g-darknet
        // https://arxiv.org/abs/1902.09630v2
        // https://giou.stanford.edu/
        all_ious.dx_iou = dx_box_iou(pred, truth, iou_loss);

        // jacobian^t (transpose)
        //float dx = (all_ious.dx_iou.dl + all_ious.dx_iou.dr);
        //float dy = (all_ious.dx_iou.dt + all_ious.dx_iou.db);
        //float dw = ((-0.5 * all_ious.dx_iou.dl) + (0.5 * all_ious.dx_iou.dr));
        //float dh = ((-0.5 * all_ious.dx_iou.dt) + (0.5 * all_ious.dx_iou.db));

        // jacobian^t (transpose)
        float dx = all_ious.dx_iou.dt;
        float dy = all_ious.dx_iou.db;
        float dw = all_ious.dx_iou.dl;
        float dh = all_ious.dx_iou.dr;

        // predict exponential, apply gradient of e^delta_t ONLY for w,h
        dw *= exp(x[index + 2 * stride]);
        dh *= exp(x[index + 3 * stride]);

        // normalize iou weight
        dx *= iou_normalizer;
        dy *= iou_normalizer;
        dw *= iou_normalizer;
        dh *= iou_normalizer;

        dx = fix_nan_inf(dx);
        dy = fix_nan_inf(dy);
        dw = fix_nan_inf(dw);
        dh = fix_nan_inf(dh);

        dx = clip_value(dx, max_delta);
        dy = clip_value(dy, max_delta);
        dw = clip_value(dw, max_delta);
        dh = clip_value(dh, max_delta);

        if (!accumulate) {
            delta[index + 0 * stride] = 0;
            delta[index + 1 * stride] = 0;
            delta[index + 2 * stride] = 0;
            delta[index + 3 * stride] = 0;
        }

        // accumulate delta
        delta[index + 0 * stride] += dx;
        delta[index + 1 * stride] += dy;
        delta[index + 2 * stride] += dw;
        delta[index + 3 * stride] += dh;
    }
    //返回梯度
    return all_ious;
}

//对梯度进行平均
void averages_yolo_deltas(int class_index, int box_index, int stride, int classes, float *delta)
{

    int classes_in_one_box = 0;
    int c;
	//在一个box里面bbox有多少个类别
    for (c = 0; c < classes; ++c) {
        if (delta[class_index + stride*c] > 0) classes_in_one_box++;
    }
	//梯度除以box中的物体类别数
    if (classes_in_one_box > 0) {
        delta[box_index + 0 * stride] /= classes_in_one_box;
        delta[box_index + 1 * stride] /= classes_in_one_box;
        delta[box_index + 2 * stride] /= classes_in_one_box;
        delta[box_index + 3 * stride] /= classes_in_one_box;
    }
}

//计算类别误差
void delta_yolo_class(float *output, float *delta, int index, int class_id, int classes, int stride, float *avg_cat, int focal_loss, float label_smooth_eps, float *classes_multipliers)
{
    int n;
    if (delta[index + stride*class_id]){ //应该不会进入这个判断，因为 delta[index] 初值为0
        delta[index + stride*class_id] = (1 - label_smooth_eps) - output[index + stride*class_id];
        if (classes_multipliers) delta[index + stride*class_id] *= classes_multipliers[class_id];
        if(avg_cat) *avg_cat += output[index + stride*class_id];
        return;
    }
    // Focal loss
    if (focal_loss) {
        // Focal Loss
        float alpha = 0.5;    // 0.25 or 0.5
        //float gamma = 2;    // hardcoded in many places of the grad-formula

        int ti = index + stride*class_id;
        float pt = output[ti] + 0.000000000000001F;
        // http://fooplot.com/#W3sidHlwZSI6MCwiZXEiOiItKDEteCkqKDIqeCpsb2coeCkreC0xKSIsImNvbG9yIjoiIzAwMDAwMCJ9LHsidHlwZSI6MTAwMH1d
        float grad = -(1 - pt) * (2 * pt*logf(pt) + pt - 1);    // http://blog.csdn.net/linmingan/article/details/77885832
        //float grad = (1 - pt) * (2 * pt*logf(pt) + pt - 1);    // https://github.com/unsky/focal-loss

        for (n = 0; n < classes; ++n) { //对所有类别，如果预测正确，则误差为 1-predict，否则为 0-predict
            delta[index + stride*n] = (((n == class_id) ? 1 : 0) - output[index + stride*n]);

            delta[index + stride*n] *= alpha*grad;

            if (n == class_id && avg_cat) *avg_cat += output[index + stride*n];
        }
    }
    else {
        // default
        for (n = 0; n < classes; ++n) {
            delta[index + stride*n] = ((n == class_id) ? (1 - label_smooth_eps) : (0 + label_smooth_eps/classes)) - output[index + stride*n];
            if (classes_multipliers && n == class_id) delta[index + stride*class_id] *= classes_multipliers[class_id];
            if (n == class_id && avg_cat) *avg_cat += output[index + stride*n];
        }
    }
}

//调用接口如下：compare_yolo_class(l.output, l.classes, class_index, l.w*l.h, objectness, class_id, 0.25f)
//获得预测bbox 的类别信息，如果某个类别的概率超过0.25返回1
int compare_yolo_class(float *output, int classes, int class_index, int stride, float objectness, int class_id, float conf_thresh)
{
    int j;
    for (j = 0; j < classes; ++j) {
		//遍历所有类别
        //float prob = objectness * output[class_index + stride*j];
		//这个stride*j是因为yolo里面数据的排布方式确定，看下面这个函数的解释就知道了
        float prob = output[class_index + stride*j];
		//大于阈值就返回1
        if (prob > conf_thresh) {
            return 1;
        }
    }
    return 0;
}

/** 
 * @brief 计算某个矩形框中某个参数在l.output中的索引。一个矩形框包含了x,y,w,h,c,C1,C2...,Cn信息，
 *        前四个用于定位，第五个为矩形框含有物体的置信度信息c，即矩形框中存在物体的概率为多大，而C1到Cn
 *        为矩形框中所包含的物体分别属于这n类物体的概率。本函数负责获取该矩形框首个定位信息也即x值在
 *        l.output中索引、获取该矩形框置信度信息c在l.output中的索引、获取该矩形框分类所属概率的首个
 *        概率也即C1值的索引，具体是获取矩形框哪个参数的索引，取决于输入参数entry的值，这些在
 *        forward_region_layer()函数中都有用到，由于l.output的存储方式，当entry=0时，就是获取矩形框
 *        x参数在l.output中的索引；当entry=4时，就是获取矩形框置信度信息c在l.output中的索引；当
 *        entry=5时，就是获取矩形框首个所属概率C1在l.output中的索引，具体可以参考forward_region_layer()
 *        中调用本函数时的注释.
 * @param l 当前region_layer
 * @param batch 当前照片是整个batch中的第几张，因为l.output中包含整个batch的输出，所以要定位某张训练图片
 *              输出的众多网格中的某个矩形框，当然需要该参数.
 * @param location 这个参数，说实话，感觉像个鸡肋参数，函数中用这个参数获取n和loc的值，这个n就是表示网格中
 *                 的第几个预测矩形框（比如每个网格预测5个矩形框，那么n取值范围就是从0~4，loc就是某个
 *                 通道上的元素偏移（region_layer输出的通道数为l.out_c = (classes + coords + 1)，
 *                 这样说可能没有说明白，这都与l.output的存储结构相关，见下面详细注释以及其他说明。总之，
 *                 查看一下调用本函数的父函数forward_region_layer()就知道了，可以直接输入n和j*l.w+i的，
 *                 没有必要输入location，这样还得重新计算一次n和loc.               
 * @param entry 切入点偏移系数，关于这个参数，就又要扯到l.output的存储结构了，见下面详细注释以及其他说明.
 * @details l.output这个参数的存储内容以及存储方式已经在多个地方说明了，再多的文字都不及图文说明，此处再
 *          简要罗嗦几句，更为具体的参考图文说明。l.output中存储了整个batch的训练输出，每张训练图片都会输出
 *          l.out_w*l.out_h个网格，每个网格会预测l.n个矩形框，每个矩形框含有l.classes+l.coords+1个参数，
 *          而最后一层的输出通道数为l.n*(l.classes+l.coords+1)，可以想象下最终输出的三维张量是个什么样子的。
 *          展成一维数组存储时，l.output可以首先分成batch个大段，每个大段存储了一张训练图片的所有输出；进一步细分，
 *          取其中第一大段分析，该大段中存储了第一张训练图片所有输出网格预测的矩形框信息，每个网格预测了l.n个矩形框，
 *          存储时，l.n个矩形框是分开存储的，也就是先存储所有网格中的第一个矩形框，而后存储所有网格中的第二个矩形框，
 *          依次类推，如果每个网格中预测5个矩形框，则可以继续把这一大段分成5个中段。继续细分，5个中段中取第
 *          一个中段来分析，这个中段中按行（有l.out_w*l.out_h个网格，按行存储）依次存储了这张训练图片所有输出网格中
 *          的第一个矩形框信息，要注意的是，这个中段存储的顺序并不是挨个挨个存储每个矩形框的所有信息，
 *          而是先存储所有矩形框的x，而后是所有的y,然后是所有的w,再是h，c，最后的的概率数组也是拆分进行存储，
 *          并不是一下子存储完一个矩形框所有类的概率，而是先存储所有网格所属第一类的概率，再存储所属第二类的概率，
 *          具体来说这一中段首先存储了l.out_w*l.out_h个x，然后是l.out_w*l.out_c个y，依次下去，
 *          最后是l.out_w*l.out_h个C1（属于第一类的概率，用C1表示，下面类似），l.out_w*l.outh个C2,...,
 *          l.out_w*l.out_c*Cn（假设共有n类），所以可以继续将中段分成几个小段，依次为x,y,w,h,c,C1,C2,...Cn
 *          小段，每小段的长度都为l.out_w*l.out_c.
 *          现在回过来看本函数的输入参数，batch就是大段的偏移数（从第几个大段开始，对应是第几张训练图片），
 *          由location计算得到的n就是中段的偏移数（从第几个中段开始，对应是第几个矩形框），
 *          entry就是小段的偏移数（从几个小段开始，对应具体是那种参数，x,c还是C1），而loc则是最后的定位，
 *          前面确定好第几大段中的第几中段中的第几小段的首地址，loc就是从该首地址往后数loc个元素，得到最终定位
 *          某个具体参数（x或c或C1）的索引值，比如l.output中存储的数据如下所示（这里假设只存了一张训练图片的输出，
 *          因此batch只能为0；并假设l.out_w=l.out_h=2,l.classes=2）：
 *          xxxxyyyywwwwhhhhccccC1C1C1C1C2C2C2C2-#-xxxxyyyywwwwhhhhccccC1C1C1C1C2C2C2C2，
 *          n=0则定位到-#-左边的首地址（表示每个网格预测的第一个矩形框），n=1则定位到-#-右边的首地址（表示每个网格预测的第二个矩形框）
 *          entry=0,loc=0获取的是x的索引，且获取的是第一个x也即l.out_w*l.out_h个网格中第一个网格中第一个矩形框x参数的索引；
 *          entry=4,loc=1获取的是c的索引，且获取的是第二个c也即l.out_w*l.out_h个网格中第二个网格中第一个矩形框c参数的索引；
 *          entry=5,loc=2获取的是C1的索引，且获取的是第三个C1也即l.out_w*l.out_h个网格中第三个网格中第一个矩形框C1参数的索引；
 *          如果要获取第一个网格中第一个矩形框w参数的索引呢？如果已经获取了其x值的索引，显然用x的索引加上3*l.out_w*l.out_h即可获取到，
 *          这正是delta_region_box()函数的做法；
 *          如果要获取第三个网格中第一个矩形框C2参数的索引呢？如果已经获取了其C1值的索引，显然用C1的索引加上l.out_w*l.out_h即可获取到，
 *          这正是delta_region_class()函数中的做法；
 *          由上可知，entry=0时,即偏移0个小段，是获取x的索引；entry=4,是获取自信度信息c的索引；entry=5，是获取C1的索引.
 *          l.output的存储方式大致就是这样，个人觉得说的已经很清楚了，但可视化效果终究不如图文说明～
*/
static int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;
}

//前向传播
void forward_yolo_layer(const layer l, network_state state)
{
    int i, j, b, t, n;
	//将层输入直接拷贝到层输出
    memcpy(l.output, state.input, l.outputs*l.batch * sizeof(float));
   //在 cpu 里，把预测输出的 x,y,confidence 和80种类别都 sigmoid 激活，确保值在0~1
#ifndef GPU
    for (b = 0; b < l.batch; ++b) {
        for (n = 0; n < l.n; ++n) {
			// 获取第b个batch开始的index
            int index = entry_index(l, b, n*l.w*l.h, 0);
			// 对预测的tx,ty进行逻辑回归预测,
            activate_array(l.output + index, 2 * l.w*l.h, LOGISTIC);        // x,y,
            scal_add_cpu(2 * l.w*l.h, l.scale_x_y, -0.5*(l.scale_x_y - 1), l.output + index, 1);    // scale x,y
            // 获取第b个batch confidence开始的index
			index = entry_index(l, b, n*l.w*l.h, 4);
			// 对预测的confidence以及class进行逻辑回归
            activate_array(l.output + index, (1 + l.classes)*l.w*l.h, LOGISTIC);
        }
    }
#endif

    // delta is zeroed
	//将yolo层的误差项进行初始化(包含整个batch的)
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
	// inference阶段,到此结束
    if (!state.train) return;
    //float avg_iou = 0;
    float tot_iou = 0; //总的IoU（Intersection over Union）
    float tot_giou = 0;
    float tot_diou = 0;
    float tot_ciou = 0;
    float tot_iou_loss = 0;
    float tot_giou_loss = 0;
    float tot_diou_loss = 0;
    float tot_ciou_loss = 0;
    float recall = 0;
    float recall75 = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0; // yolo层的总损失初始化为0
    for (b = 0; b < l.batch; ++b) {// 遍历batch中的每一张图片
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {// 遍历每个cell, 当前cell编号[j, i]
                for (n = 0; n < l.n; ++n) {// 遍历每一个bbox, 当前bbox编号 [n]
					// 在这里与yolov2 reorg层是相似的, 获得第j*w+i个cell第n个bbox的index
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
					// 计算第j*w+i个cell第n个bbox在当前特征图上的相对位置[x,y],在网络输入图片上的相对宽度,高度[w,h]
                    box pred = get_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.w*l.h);
                    float best_match_iou = 0;
                    int best_match_t = 0;
                    float best_iou = 0; // 保存最大iou
                    int best_t = 0;// 保存最大iou的bbox id
                    for (t = 0; t < l.max_boxes; ++t) {// 遍历每一个GT bbox
						// 将第t个bbox由float数组转bbox结构体,方便计算iou
                        box truth = float_to_box_stride(state.truth + t*(4 + 1) + b*l.truths, 1);
						//获取第t个bbox的类别，检查是否有标注错误
                        int class_id = state.truth[t*(4 + 1) + b*l.truths + 4];
                        if (class_id >= l.classes) {
                            printf(" Warning: in txt-labels class_id=%d >= classes=%d in cfg-file. In txt-labels class_id should be [from 0 to %d] \n", class_id, l.classes, l.classes - 1);
                            printf(" truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f, class_id = %d \n", truth.x, truth.y, truth.w, truth.h, class_id);
                            getchar();
                            continue; // if label contains class_id more than number of classes in the cfg-file
                        }
						// 如果x坐标为0则取消,因为yolov3这里定义了max_boxes个bbox
                        if (!truth.x) break;  // continue;

                        int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);//预测bbox 类别s下标
                        int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4); //预测bbox objectness下标
                        float objectness = l.output[obj_index]; //预测bbox object置信度
						//获得预测bbox 的类别信息，如果某个类别的概率超过0.25返回1
                        int class_id_match = compare_yolo_class(l.output, l.classes, class_index, l.w*l.h, objectness, class_id, 0.25f);

                        float iou = box_iou(pred, truth); // 计算pred bbox与第t个GT bbox之间的iou
						// 这个地方和原始的DarkNet实现不太一样，多了一个class_id_match=1的限制，即预测bbox的置信度必须大于0.25
                        if (iou > best_match_iou && class_id_match == 1) {
                            best_match_iou = iou;
                            best_match_t = t;
                        }
                        if (iou > best_iou) {
                            best_iou = iou; // 记录iou最大的iou
                            best_t = t; // 记录该GT bbox的编号t
                        }
                    }
					// 在这里与yolov2 reorg层是相似的, 获得第j*w+i个cell第n个bbox的confidence
                    int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);
					// 统计pred bbox的confidence
                    avg_anyobj += l.output[obj_index];
					 // 与yolov1相似,先将所有pred bbox都当做noobject, 计算其confidence梯度，不过这里多了一个平衡系数
                    l.delta[obj_index] = l.cls_normalizer * (0 - l.output[obj_index]);
					// best_iou大于阈值则说明pred box有物体,在yolov3中正样本阈值ignore_thresh=.5
                    if (best_match_iou > l.ignore_thresh) {
                        l.delta[obj_index] = 0;
                    }
					// pred bbox为完全预测正确样本,在yolov3完全预测正确样本的阈值truth_thresh=1.
					//这个参数在cfg文件中，值为1，这个条件语句永远不可能成立
                    if (best_iou > l.truth_thresh) {
						// 作者在YOLOV3论文中的第4节提到了这部分。
						// 作者尝试Faster-RCNN中提到的双IOU策略，当Anchor与GT的IoU大于0.7时，该Anchor被算作正样本
						//计入损失中，但训练过程中并没有产生好的结果，所以最后放弃了。
                        l.delta[obj_index] = l.cls_normalizer * (1 - l.output[obj_index]);
						 // 获得best_iou对应GT bbox的class的index
                        int class_id = state.truth[best_t*(4 + 1) + b*l.truths + 4];
						//yolov3 yolo层中map=0, 不参与计算
                        if (l.map) class_id = l.map[class_id]; 
						// 获得best_iou对应pred bbox的class的index
                        int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);
                        delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w*l.h, 0, l.focal_loss, l.label_smooth_eps, l.classes_multipliers);
                        box truth = float_to_box_stride(state.truth + best_t*(4 + 1) + b*l.truths, 1);
                        const float class_multiplier = (l.classes_multipliers) ? l.classes_multipliers[class_id] : 1.0f;
                        // 计算pred bbox的[x,y,w,h]的梯度
						delta_yolo_box(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2 - truth.w*truth.h), l.w*l.h, l.iou_normalizer * class_multiplier, l.iou_loss, 1, l.max_delta);
                    }
                }
            }
        }
        for (t = 0; t < l.max_boxes; ++t) {
			//遍历每一个GT box
			// 将第t个bbox由float数组转bbox结构体,方便计算iou
            box truth = float_to_box_stride(state.truth + t*(4 + 1) + b*l.truths, 1);
            if (truth.x < 0 || truth.y < 0 || truth.x > 1 || truth.y > 1 || truth.w < 0 || truth.h < 0) {
                char buff[256];
                printf(" Wrong label: truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f \n", truth.x, truth.y, truth.w, truth.h);
                sprintf(buff, "echo \"Wrong label: truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f\" >> bad_label.list",
                    truth.x, truth.y, truth.w, truth.h);
                system(buff);
            }
            int class_id = state.truth[t*(4 + 1) + b*l.truths + 4];
            if (class_id >= l.classes) continue; // if label contains class_id more than number of classes in the cfg-file

            if (!truth.x) break;  // 如果x坐标为0则取消，因为yolov3定义了max_boxes个bbox,可能实际上没那么多
            float best_iou = 0; //保存最大的IOU
            int best_n = 0; //保存最大IOU的bbox index
            i = (truth.x * l.w); // 获得当前t个GT bbox所在的cell
            j = (truth.y * l.h); 
            box truth_shift = truth;
            truth_shift.x = truth_shift.y = 0; //将truth_shift的box位置移动到0,0
            for (n = 0; n < l.total; ++n) { // 遍历每一个anchor bbox找到与GT bbox最大的IOU
                box pred = { 0 };
                pred.w = l.biases[2 * n] / state.net.w; // 计算pred bbox的w在相对整张输入图片的位置
                pred.h = l.biases[2 * n + 1] / state.net.h; // 计算pred bbox的h在相对整张输入图片的位置
                float iou = box_iou(pred, truth_shift); // 计算GT box truth_shift 与 预测bbox pred二者之间的IOU
                if (iou > best_iou) { 
                    best_iou = iou;// 记录最大的IOU
                    best_n = n;// 以及记录该bbox的编号n
                }
            }
            // 上面记录bbox的编号,是否由该层Anchor预测的
            int mask_n = int_index(l.mask, best_n, l.n);
            if (mask_n >= 0) {
                int class_id = state.truth[t*(4 + 1) + b*l.truths + 4];
                if (l.map) class_id = l.map[class_id];
				// 获得best_iou对应anchor box的index
                int box_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 0);
				//这个参数是用来控制样本数量不均衡的，即Focal Loss中的alpha
                const float class_multiplier = (l.classes_multipliers) ? l.classes_multipliers[class_id] : 1.0f;
				// 计算best_iou对应Anchor bbox的[x,y,w,h]的梯度
                ious all_ious = delta_yolo_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2 - truth.w*truth.h), l.w*l.h, l.iou_normalizer * class_multiplier, l.iou_loss, 1, l.max_delta);

				// 下面的都是模板检测最新的工作，metricl learning，包括IOU/GIOU/DIOU/CIOU Loss
                // range is 0 <= 1
                tot_iou += all_ious.iou;
                tot_iou_loss += 1 - all_ious.iou;
                // range is -1 <= giou <= 1
                tot_giou += all_ious.giou;
                tot_giou_loss += 1 - all_ious.giou;

                tot_diou += all_ious.diou;
                tot_diou_loss += 1 - all_ious.diou;

                tot_ciou += all_ious.ciou;
                tot_ciou_loss += 1 - all_ious.ciou;
				// 获得best_iou对应anchor box的confidence的index
                int obj_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4);
				//统计confidence
                avg_obj += l.output[obj_index];
				// 计算confidence的梯度
                l.delta[obj_index] = class_multiplier * l.cls_normalizer * (1 - l.output[obj_index]);
				// 获得best_iou对应GT box的class的index
                int class_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + 1);
				// 获得best_iou对应anchor box的class的index
                delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w*l.h, &avg_cat, l.focal_loss, l.label_smooth_eps, l.classes_multipliers);

                ++count;
                ++class_count;
                if (all_ious.iou > .5) recall += 1;
                if (all_ious.iou > .75) recall75 += 1;
            }

			//下面这个过程和上面一样，不过多约束了一个iou_thresh
            // iou_thresh
            for (n = 0; n < l.total; ++n) {
                int mask_n = int_index(l.mask, n, l.n);
                if (mask_n >= 0 && n != best_n && l.iou_thresh < 1.0f) {
                    box pred = { 0 };
                    pred.w = l.biases[2 * n] / state.net.w;
                    pred.h = l.biases[2 * n + 1] / state.net.h;
                    float iou = box_iou(pred, truth_shift);
                    // iou, n

                    if (iou > l.iou_thresh) {
                        int class_id = state.truth[t*(4 + 1) + b*l.truths + 4];
                        if (l.map) class_id = l.map[class_id];

                        int box_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 0);
                        const float class_multiplier = (l.classes_multipliers) ? l.classes_multipliers[class_id] : 1.0f;
                        ious all_ious = delta_yolo_box(truth, l.output, l.biases, n, box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2 - truth.w*truth.h), l.w*l.h, l.iou_normalizer * class_multiplier, l.iou_loss, 1, l.max_delta);

                        // range is 0 <= 1
                        tot_iou += all_ious.iou;
                        tot_iou_loss += 1 - all_ious.iou;
                        // range is -1 <= giou <= 1
                        tot_giou += all_ious.giou;
                        tot_giou_loss += 1 - all_ious.giou;

                        tot_diou += all_ious.diou;
                        tot_diou_loss += 1 - all_ious.diou;

                        tot_ciou += all_ious.ciou;
                        tot_ciou_loss += 1 - all_ious.ciou;

                        int obj_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4);
                        avg_obj += l.output[obj_index];
                        l.delta[obj_index] = class_multiplier * l.cls_normalizer * (1 - l.output[obj_index]);

                        int class_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + 1);
                        delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w*l.h, &avg_cat, l.focal_loss, l.label_smooth_eps, l.classes_multipliers);

                        ++count;
                        ++class_count;
                        if (all_ious.iou > .5) recall += 1;
                        if (all_ious.iou > .75) recall75 += 1;
                    }
                }
            }
        }

        // averages the deltas obtained by the function: delta_yolo_box()_accumulate
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
					// 在这里与yolov2 reorg层是相似的, 获得第j*w+i个cell第n个bbox的index
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
					//获得第j*w+i个cell第n个bbox的类别
                    int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);
					//特征图的大小
                    const int stride = l.w*l.h;
					//对梯度进行平均
                    averages_yolo_deltas(class_index, box_index, stride, l.classes, l.delta);
                }
            }
        }
    }

    //*(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    //printf("Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n", state.index, avg_iou / count, avg_cat / class_count, avg_obj / count, avg_anyobj / (l.w*l.h*l.n*l.batch), recall / count, recall75 / count, count);

    int stride = l.w*l.h;
    float* no_iou_loss_delta = (float *)calloc(l.batch * l.outputs, sizeof(float));
    memcpy(no_iou_loss_delta, l.delta, l.batch * l.outputs * sizeof(float));
    for (b = 0; b < l.batch; ++b) {
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
					//yolov3如果使用的是iou loss，也就是metric learning的方式，那么x,y,w,h的loss可以设置为0
                    int index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    no_iou_loss_delta[index + 0 * stride] = 0;
                    no_iou_loss_delta[index + 1 * stride] = 0;
                    no_iou_loss_delta[index + 2 * stride] = 0;
                    no_iou_loss_delta[index + 3 * stride] = 0;
                }
            }
        }
    }
	//计算所有的分类loss
    float classification_loss = l.cls_normalizer * pow(mag_array(no_iou_loss_delta, l.outputs * l.batch), 2);
    free(no_iou_loss_delta);
	//计算总的loss
    float loss = pow(mag_array(l.delta, l.outputs * l.batch), 2);
	//计算回归loss
    float iou_loss = loss - classification_loss;

    float avg_iou_loss = 0;
    // gIOU loss + MSE (objectness) loss
    if (l.iou_loss == MSE) {
        *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    }
    else {
        // Always compute classification loss both for iou + cls loss and for logging with mse loss
        // TODO: remove IOU loss fields before computing MSE on class
        //   probably split into two arrays
        if (l.iou_loss == GIOU) {
            avg_iou_loss = count > 0 ? l.iou_normalizer * (tot_giou_loss / count) : 0;
        }
        else {
			//count代表目标个数
            avg_iou_loss = count > 0 ? l.iou_normalizer * (tot_iou_loss / count) : 0;
        }
        *(l.cost) = avg_iou_loss + classification_loss;
    }

    loss /= l.batch;
    classification_loss /= l.batch;
    iou_loss /= l.batch;

    printf("v3 (%s loss, Normalizer: (iou: %f, cls: %f) Region %d Avg (IOU: %f, GIOU: %f), Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f, count: %d, loss = %f, class_loss = %f, iou_loss = %f\n",
        (l.iou_loss == MSE ? "mse" : (l.iou_loss == GIOU ? "giou" : "iou")), l.iou_normalizer, l.cls_normalizer, state.index, tot_iou / count, tot_giou / count, avg_cat / class_count, avg_obj / count, avg_anyobj / (l.w*l.h*l.n*l.batch), recall / count, recall75 / count, count,
        loss, classification_loss, iou_loss);
}

//误差反向传播
void backward_yolo_layer(const layer l, network_state state)
{
	//直接把 l.delta 拷贝给上一层的 delta。注意 net.delta 指向 prev_layer.delta。
   axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
}

// Converts output of the network to detection boxes
// w,h: image width,height
// netw,neth: network width,height
// relative: 1 (all callers seems to pass TRUE)
//调整预测 box 中心和大小

void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative, int letter)
{
	//w 和 h 是输入图片的尺寸，netw 和 neth 是网络输入尺寸
    int i;
    // network height (or width)
    int new_w = 0;
    // network height (or width)
    int new_h = 0;
    // Compute scale given image w,h vs network w,h
    // I think this "rotates" the image to match network to input image w/h ratio
    // new_h and new_w are really just network width and height
    if (letter) {
        if (((float)netw / w) < ((float)neth / h)) { //新图片尺寸
            new_w = netw;
            new_h = (h * netw) / w;
        }
        else {
            new_h = neth;
            new_w = (w * neth) / h;
        }
    }
    else {
        new_w = netw;
        new_h = neth;
    }
    // difference between network width and "rotated" width
    float deltaw = netw - new_w;
    // difference between network height and "rotated" height
    float deltah = neth - new_h;
    // ratio between rotated network width and network width
    float ratiow = (float)new_w / netw;
    // ratio between rotated network width and network width
    float ratioh = (float)new_h / neth;
    for (i = 0; i < n; ++i) { //调整 box 相对新图片尺寸的位置

        box b = dets[i].bbox;
        // x = ( x - (deltaw/2)/netw ) / ratiow;
        //   x - [(1/2 the difference of the network width and rotated width) / (network width)]
        b.x = (b.x - deltaw / 2. / netw) / ratiow;
        b.y = (b.y - deltah / 2. / neth) / ratioh;
        // scale to match rotation of incoming image
        b.w *= 1 / ratiow;
        b.h *= 1 / ratioh;

        // relative seems to always be == 1, I don't think we hit this condition, ever.
        if (!relative) {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }

        dets[i].bbox = b;
    }
}

/*
void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative, int letter)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (letter) {
        if (((float)netw / w) < ((float)neth / h)) {
            new_w = netw;
            new_h = (h * netw) / w;
        }
        else {
            new_h = neth;
            new_w = (w * neth) / h;
        }
    }
    else {
        new_w = netw;
        new_h = neth;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw);
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth);
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}
*/

//预测输出中置信度超过阈值的 box 个数
int yolo_num_detections(layer l, float thresh)
{
    int i, n;
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
			////获得置信度偏移位置
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
			//置信度超过阈值
            if(l.output[obj_index] > thresh){
                ++count;
            }
        }
    }
    return count;
}

void avg_flipped_yolo(layer l)
{
    int i,j,n,z;
    float *flip = l.output + l.outputs;
    for (j = 0; j < l.h; ++j) {
        for (i = 0; i < l.w/2; ++i) {
            for (n = 0; n < l.n; ++n) {
                for(z = 0; z < l.classes + 4 + 1; ++z){
                    int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                    int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                    float swap = flip[i1];
                    flip[i1] = flip[i2];
                    flip[i2] = swap;
                    if(z == 0){
                        flip[i1] = -flip[i1];
                        flip[i2] = -flip[i2];
                    }
                }
            }
        }
    }
    for(i = 0; i < l.outputs; ++i){
        l.output[i] = (l.output[i] + flip[i])/2.;
    }
}

//获得预测输出中超过阈值的 box
int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter)
{
    //printf("\n l.batch = %d, l.w = %d, l.h = %d, l.n = %d \n", l.batch, l.w, l.h, l.n);
    int i,j,n;
    float *predictions = l.output;
    // This snippet below is not necessary
    // Need to comment it in order to batch processing >= 2 images
    //if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            float objectness = predictions[obj_index]; //置信度
            //if(objectness <= thresh) continue;    // incorrect behavior for Nan values
            if (objectness > thresh) {
                //printf("\n objectness = %f, thresh = %f, i = %d, n = %d \n", objectness, thresh, i, n);
                int box_index = entry_index(l, 0, n*l.w*l.h + i, 0);
                dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
                dets[count].objectness = objectness;
                dets[count].classes = l.classes;
                for (j = 0; j < l.classes; ++j) {
                    int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                    float prob = objectness*predictions[class_index];//置信度 x 类别概率
                    dets[count].prob[j] = (prob > thresh) ? prob : 0;//小于阈值则概率置0
                }
                ++count;
            }
        }
    }
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative, letter);//调整 box 大小
    return count;
}

#ifdef GPU

void forward_yolo_layer_gpu(const layer l, network_state state)
{
    //copy_ongpu(l.batch*l.inputs, state.input, 1, l.output_gpu, 1);
    simple_copy_ongpu(l.batch*l.inputs, state.input, l.output_gpu);
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            // y = 1./(1. + exp(-x))
            // x = ln(y/(1-y))  // ln - natural logarithm (base = e)
            // if(y->1) x -> inf
            // if(y->0) x -> -inf
            activate_array_ongpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);    // x,y
            if (l.scale_x_y != 1) scal_add_ongpu(2 * l.w*l.h, l.scale_x_y, -0.5*(l.scale_x_y - 1), l.output_gpu + index, 1);      // scale x,y
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array_ongpu(l.output_gpu + index, (1+l.classes)*l.w*l.h, LOGISTIC); // classes and objectness
        }
    }
    if(!state.train || l.onlyforward){
        //cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        cuda_pull_array_async(l.output_gpu, l.output, l.batch*l.outputs);
        CHECK_CUDA(cudaPeekAtLastError());
        return;
    }

    float *in_cpu = (float *)xcalloc(l.batch*l.inputs, sizeof(float));
    cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
    memcpy(in_cpu, l.output, l.batch*l.outputs*sizeof(float));
    float *truth_cpu = 0;
    if (state.truth) {
        int num_truth = l.batch*l.truths;
        truth_cpu = (float *)xcalloc(num_truth, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, num_truth);
    }
    network_state cpu_state = state;
    cpu_state.net = state.net;
    cpu_state.index = state.index;
    cpu_state.train = state.train;
    cpu_state.truth = truth_cpu;
    cpu_state.input = in_cpu;
    forward_yolo_layer(l, cpu_state);
    //forward_yolo_layer(l, state);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
    free(in_cpu);
    if (cpu_state.truth) free(cpu_state.truth);
}

void backward_yolo_layer_gpu(const layer l, network_state state)
{
    axpy_ongpu(l.batch*l.inputs, 1, l.delta_gpu, 1, state.delta, 1);
}
#endif
