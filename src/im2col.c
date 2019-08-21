#include "im2col.h"
#include <stdio.h>
/*
** 从输入的多通道数组im（存储图像数据）中获取指定行，列，通道数处的元素值
** im:  函数的输入，所有的数据存成一个一维数组
** height: 每一个通道的高度(即是输入图像的真正高度，补0之前)
** width: 每一个通道的宽度(即是输入图像的真正宽度，补0之前)
** channles： 输入通道数
** row: 要提取的元素所在的行(padding之后的行数)
** col: 要提取的元素所在的列(padding之后的列数)
** channel: 要提取的元素所在的通道
** pad: 图像上下左右补0的个数，四周是一样的
** 返回im中channel通道，row-pad行,col-pad列处的元素值
*/
float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) 
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}

