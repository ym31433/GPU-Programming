#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <math.h>

#define BLOCKSIZE 525
#define WORDLENGTH 500
__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

/*__device__ int findNextUp(int idx, int& lastStep) {
    if(idx%2) {
        --idx; //from right
        lastStep = 1;
    }
    else {
        idx = idx/2-1; //from child
        lastStep = 0;
    }
    return idx;
}*/

__global__ void BuildTree_CountPosition(const char* text, int* pos, int text_size) {
    //build trees
    int N;
    if(blockIdx.x == 0) {
        N = blockDim.x;
        __shared__ int tree[2*BLOCKSIZE-1];
    }
    else {
        N = blockDim.x+(WORDLENGTH-1); //number of the bottom nodes of the tree
        __shared__ int tree[2*(BLOCKSIZE+WORDLENGTH-1)-1];
    }
    //cudaMalloc((void**)&tree, sizeof(int)*(N*2-1));
    int textIdx = threadIdx.x + blockIdx.x*blockDim.x;
    int treeIdx = (blockIdx.x == 0)? threadIdx.x+(N-1): threadIdx.x+(N-1)+(WORDLENGTH-1);
    //initialize the bottom of the tree
    tree[treeIdx] = (text[textIdx] == '\n')? 0: 1;
    if(blockIdx.x != 0 && threadIdx.x < (WORDLENGTH-1)) tree[treeIdx-(WORDLENGTH-1)] = (text[textIdx-(WORDLENGTH-1)] == '\n')? 0: 1;
    __syncthreads();
    //build the upper tree
    int parent, leftChild, rightChild;
    for(int p = N/2-1; p > 0; p = (p-1)/2 ) { //not including root
        parent = threadIdx.x+p;
        leftChild = parent*2+1;
        rightChild = parent*2+2;
        if(threadIdx.x <= p) {
            tree[parent] = ( (tree[leftChild] & tree[rightChild]) == 0)? 0: tree[leftChild]+tree[rightChild];
        }
        __syncthreads();
    }
    //root
    if((tree[1] & tree[2]) == 0) tree[0] = 0;
    else tree[0] = tree[1] + tree[2];
    //count position
    bool lastStep = 1;// 1 from right, 0 from child
    int idx = treeIdx;
    //count up
    while(idx >= 0 && tree[idx] != 0) {
        if(lastStep) pos[textIdx] += tree[idx];//from right
        else pos[textIdx] += (0.5*tree[idx]);//from child
        if( ( (idx+1) & -(idx+1) ) == (idx+1) ) break;//if the node is at left most side
        //find next
        if(idx%2) {
            --idx; //from right
            lastStep = 1;
        }
        else {
            idx = idx/2-1; //from child
            lastStep = 0;
        }
    }
    //count down
    if( ( (idx+1) & -(idx+1) ) == (idx+1) && tree[idx] != 0) return;//if the node is at left most side and is not zero
    if(lastStep) idx = idx*2+2;//from right, next is right child
    else idx = idx*2+1;//from child
    while( idx > 0 && idx < N/2-1 && tree[idx] == 0) {
        idx = idx*2+2;
    }
    pos[textIdx]+=tree[idx];
}

void CountPosition(const char *text, int *pos, int text_size)
{
    //try to let gpu build trees
    //int N = text_size/512+1;//number of trees

    /*fool!!!don't need to pass array into gpu!!!!
    MemoryBuffer<int> tree[N];
    auto tree_sync[N];
    int* tree_gpu[N];
    for(int i = 0; i != N; ++i) {
        cudaMalloc(&tree[i], sizeof(int)*1023); 
        tree_sync[i] = tree[i].CreateSync(1023);
        tree_gpu[i] = tree_sync.get_gpu_rw();
        cudaMemset(tree_gpu[i], 0, sizeof(int)*1023);
    }*/
    BuildTree_CountPosition<<<(text_size+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE>>>(text, pos, text_size);
    
    //int* tree_cpu = tree_sync.get_cpu_rw();
    /*for(int i = text_size; i != text_size*2; ++i) {
        tree_cpu[i] = (text[i-text_size] == 32)? 0: 1;
    }
    for(int i = text_size; i != 0; --i) {
        if(tree_cpu[])
        tree_cpu[i-1] = (tree_)
    }*/

    
}

int ExtractHead(const int *pos, int *head, int text_size)
{
	int *buffer;
	int nhead;
	cudaMalloc(&buffer, sizeof(int)*text_size*2); // this is enough
	thrust::device_ptr<const int> pos_d(pos);
	thrust::device_ptr<int> head_d(head), flag_d(buffer), cumsum_d(buffer+text_size);

	// TODO
/*    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int headIdx = 0;
    if(idx < text_size ) {
        if(pos[idx] == 1) {
            head[headIdx++] = idx; 
        }
    } 
*/
    nhead = 0;

//do not touch this
	cudaFree(buffer);
	return nhead;
}

void Part3(char *text, int *pos, int *head, int text_size, int n_head)
{
}
