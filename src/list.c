#include <stdlib.h>
#include <string.h>
#include "list.h"
#include "utils.h"
#include "option_list.h"

//初始化链表
list *make_list()
{
    list* l = (list*)xmalloc(sizeof(list));
    l->size = 0;
    l->front = 0;
    l->back = 0;
    return l;
}

/*
void transfer_node(list *s, list *d, node *n)
{
    node *prev, *next;
    prev = n->prev;
    next = n->next;
    if(prev) prev->next = next;
    if(next) next->prev = prev;
    --s->size;
    if(s->front == n) s->front = next;
    if(s->back == n) s->back = prev;
}
*/

//链表的删除操作，删除尾指针所指节点，即最后一个节点
void *list_pop(list *l){
	// 如果链表为空
    if(!l->back) return 0;
	// node *b 指向最后一个节点
    node *b = l->back;
    void *val = b->val;
	// 更新尾指针，尾指针指向b的前一节点，即倒数第二节点。
    l->back = b->prev;
	// 如果倒数第二节点存在
    if(l->back) l->back->next = 0; //尾指针next置为null
    // 释放最后一个节点的存储空间
	free(b);
    --l->size; // 链表长度 -1

    return val;// 返回被删除节点保存的值
}

/*
 * 简介: 将 val 指针插入 list 结构体 l 中，这里相当于是用 C 实现了 C++ 中的 
 *         list 的元素插入功能
 * 
 * 参数: l    链表指针
 *         val  链表节点的元素值
 * 
 * 流程： list 中保存的是 node 指针. 因此，需要用 node 结构体将 val 包裹起来后才可以
 *       插入 list 指针 l 中
 * 
 * 注意: 此函数类似 C++ 的 insert() 插入方式；
 *      而 opion_insert() 函数类似 C++ map 的按值插入方式，比如 map[key]= value
 *      
 *      两个函数操作对象都是 list 变量， 只是操作方式略有不同。
*/
void list_insert(list *l, void *val)
{
    node* newnode = (node*)xmalloc(sizeof(node));
    newnode->val = val;
    newnode->next = 0;
    // 如果 list 的 back 成员为空(初始化为 0), 说明 l 到目前为止，还没有存入数据  
    // 另外, 令 l 的 front 为 new （此后 front 将不会再变，除非删除） 
    if(!l->back){
        l->front = newnode;
        newnode->prev = 0;
    }else{
        l->back->next = newnode;
        newnode->prev = l->back;
    }
    l->back = newnode;
    ++l->size;
}
// 释放链表节点的存储空间，从节点n开始释放，一直释放到最后一个节点。
void free_node(node *n)
{
    node *next;
    while(n) {
        next = n->next; // 获取下一节点地址
        free(n); // 释放当前节点存储空间
        n = next; //更新n，n指向下一节点
    } 
}


void free_list_val(list *l)
{
    node *n = l->front;
    node *next;
    while (n) {
        next = n->next;
        free(n->val);
        n = next;
    }
}

// 释放整个链表l的存储空间
void free_list(list *l)
{
    free_node(l->front);
    free(l);
}

// 对链表所有节点的值【节点中值为void 类型指针】 的存储空间释放
void free_list_contents(list *l)
{
    node *n = l->front;
    while(n){
        free(n->val);
        n = n->next;
    }
}

void free_list_contents_kvp(list *l)
{
    node *n = l->front;
    while (n) {
        kvp* p = (kvp*)n->val;
        free(p->key);
        free(n->val);
        n = n->next;
    }
}

// 二维指针，这里的操作是将链表l中所有节点的值进行保存，
// 因为每个节点里保存的值是 void类型的指针，故指针的指针，即二维指针
void **list_to_array(list *l)
{
	//　分配存储空间，长度l-size, 每个空间大小为一个void类型指针
    void** a = (void**)xcalloc(l->size, sizeof(void*));
    int count = 0;
    node *n = l->front; // 工作指针n指向头节点
    while(n){
		// 将工作指针n指向节点的值保存到a中
        a[count++] = n->val;
        n = n->next; // 更新工作指针
    }
    return a;
}
