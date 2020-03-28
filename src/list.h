#ifndef LIST_H
#define LIST_H

// 链表上的节点
typedef struct node{
    void *val; //前节点的内容是一个void类型的空指针
    struct node *next; //指向当前节点的下一节点
    struct node *prev; //指向当前节点的上一节点
} node;

//双向链表
typedef struct list{
    int size; //list的所有节点个数
    node *front; //list的首节点
    node *back; //list的普通节点
} list;

#ifdef __cplusplus
extern "C" {
#endif
list *make_list();
list *make_list(); // 初始化链表
// 按值查找，注意这里的值是 void类型空指针, list.c未定义，
int list_find(list *l, void *val);

void list_insert(list *, void *);

void **list_to_array(list *l);

void free_list_val(list *l);
void free_list(list *l);
void free_list_contents(list *l);
void free_list_contents_kvp(list *l);

#ifdef __cplusplus
}
#endif
#endif
