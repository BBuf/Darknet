#ifndef UTILS_H
#define UTILS_H
#include "darknet.h"
#include "list.h"

#include <stdio.h>
#include <time.h>

#ifndef M_PI
#define M_PI       3.14159265358979323846   // pi
#endif

#ifdef __cplusplus
extern "C" {
#endif

LIB_API void free_ptrs(void **ptrs, int n); // 释放二维指针的内存空间
LIB_API void top_k(float *a, int n, int k, int *index);

void *xmalloc(size_t size);
void *xcalloc(size_t nmemb, size_t size);
void *xrealloc(void *ptr, size_t size);
double what_time_is_it_now(); //获取系统当前时间
int *read_map(char *filename); 
void shuffle(void *arr, size_t n, size_t size);
void sorta_shuffle(void *arr, size_t n, size_t size, size_t sections); //对每一个sections进行洗牌操作
char *basecfg(char *cfgfile);
int alphanum_to_int(char c); // 字符转整数
char int_to_alphanum(int i); //整数转字符
int read_int(int fd); // 从文件fd中读取一个整数
void write_int(int fd, int n); // 向文件fd中写入一个整数
void read_all(int fd, char *buffer, size_t bytes); // 将内存中buffer起始的bytes个字节数据写入到fd文件中，一次性操作，若写不完，则报错
void write_all(int fd, char *buffer, size_t bytes); // 从fd文件中，读取bytes个字节数据到内存中buffer起始，一次性操作，若读不完，则报错
int read_all_fail(int fd, char *buffer, size_t bytes); // 从fd文件中，读取bytes个字节数据到内存中buffer起始，直到读完；
int write_all_fail(int fd, char *buffer, size_t bytes); // 将内存中buffer起始的bytes个字节数据写入到fd文件中，直到写完
LIB_API void find_replace(const char* str, char* orig, char* rep, char* output); // 判断str中是否出现子串，若出现则进行替代；
void replace_image_to_label(const char* input_path, char* output_path);
void error(const char *s);
void malloc_error(); // 报错，内存分配出错
void calloc_error();
void realloc_error();
void file_error(char *s); // 报错，不能打开指定目录的文件
void strip(char *s); // 过滤掉字符数组中' '和'\t'以及'\n'三种字符
void strip_args(char *s);//过滤掉参数中的指定字符
void strip_char(char *s, char bad);  //过滤掉字符数组存在的指定字符 bad
list *split_str(char *s, char delim); // 字符串切割，按照指定字符delim进行切割；
char *fgetl(FILE *fp); // 读取指定文件中的一行字符；
list *parse_csv_line(char *line); 
char *copy_string(char *s); //字符串拷贝操作
int count_fields(char *line); // 统计字符数组中有多少个 空格字符和','字符；
float *parse_fields(char *line, int n);  // 解析字符数组中的float实数
void normalize_array(float *a, int n); //对向量进行归一化操作
void scale_array(float *a, int n, float s); //对向量进行缩放操作
void translate_array(float *a, int n, float s); // 对向量进行平移操作，
int max_index(float *a, int n); //获取向量中最大值的下标
int top_max_index(float *a, int n, int k); //获取向量中第k大值得下表
float constrain(float min, float max, float a);  // 判断小数a 与 区间小数[min, max]的关系，返回相应的值
int constrain_int(int a, int min, int max); // 判断整数a 与 区间整数[min, max]的关系，返回相应的值
float mse_array(float *a, int n); 
float rand_normal();
size_t rand_size_t();
float rand_uniform(float min, float max);
float rand_scale(float s);  //随机采样的基础上，50%的概率返回S，50%的概率返回1/s
int rand_int(int min, int max); //返回一个区间[min, max]内的一个整数
float sum_array(float *a, int n);
float mean_array(float *a, int n);
void mean_arrays(float **a, int n, int els, float *avg); //求二维float数组，每一列的平均值，保存在avg一维float数组中
float variance_array(float *a, int n);
float mag_array(float *a, int n);
float mag_array_skip(float *a, int n, int * indices_to_skip);
float dist_array(float *a, float *b, int n, int sub); // 当sub为1的时候，其实就是计算两个一维数组的欧式距离
float **one_hot_encode(float *a, int n, int k); // 进行one_hot 编码，根据，float数组a中的元素进行编码
float sec(clock_t clocks); // 获取秒数
int find_int_arg(int argc, char **argv, char *arg, int def);
float find_float_arg(int argc, char **argv, char *arg, float def);
int find_arg(int argc, char* argv[], char *arg);
char *find_char_arg(int argc, char **argv, char *arg, char *def);
int sample_array(float *a, int n);
int sample_array_custom(float *a, int n);
void print_statistics(float *a, int n); // 计算向量的mean和var
unsigned int random_gen_fast(void);
float random_float_fast();
int rand_int_fast(int min, int max);
unsigned int random_gen();
float random_float();
float rand_uniform_strong(float min, float max);
float rand_precalc_random(float min, float max, float random_part);
double double_rand(void);
unsigned int uint_rand(unsigned int less_than);
int check_array_is_nan(float *arr, int size);
int check_array_is_inf(float *arr, int size);
int int_index(int *a, int val, int n); // 一维数组进行查找操作，若查找指定值，返回index，否则，返回-1；
int *random_index_order(int min, int max);
int max_int_index(int *a, int n);
boxabs box_to_boxabs(const box* b, const int img_w, const int img_h, const int bounds_check);
int make_directory(char *path, int mode);

#define max_val_cmp(a,b) (((a) > (b)) ? (a) : (b))
#define min_val_cmp(a,b) (((a) < (b)) ? (a) : (b))

#ifdef __cplusplus
}
#endif

#endif
