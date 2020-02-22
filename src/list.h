#ifndef LIST_H
#define LIST_H

// 链表上的节点
typedef struct node{
    void *val;
    struct node *next;
    struct node *prev;
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
