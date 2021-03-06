#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <math.h>
#include <iostream>
using namespace std;

#define BLOCKSIZE 524
#define WORDLENGTH 500
#define K WORDLENGTH
#define threadNum BLOCKSIZE
__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void CountPosition_slow(const char* text, int* pos) {
    int textIdx = threadIdx.x + blockIdx.x*blockDim.x;
    int result = pos[textIdx];
    while(textIdx >= 0 && text[textIdx] != '\n') {
        result++;
        textIdx--;
    }
    pos[ threadIdx.x + blockIdx.x*blockDim.x] = result;
}

__global__ void BuildTree_CountPosition(const char* text, int* pos) {
//__global__ void BuildTree_CountPosition(const char* text, int* pos, int* debug_tree) {
    //build trees
    int N = blockDim.x + WORDLENGTH-1;  //N has to be assigned blockDim.x not BLOCKSIZE!!!!!!
    __shared__ int tree[(WORDLENGTH+BLOCKSIZE-1)*2-1];
    int textIdx = threadIdx.x + blockIdx.x*blockDim.x;
    int treeIdx = threadIdx.x+(N-1)+(WORDLENGTH-1);
    //initialize the bottom of the tree
    tree[treeIdx] = (text[textIdx] == '\n')? 0: 1;
//if(blockIdx.x == 1) debug_tree[treeIdx] = tree[treeIdx];
    if(threadIdx.x < (WORDLENGTH-1)) {
        if(blockIdx.x != 0) {
            tree[treeIdx-(WORDLENGTH-1)] = (text[textIdx-(WORDLENGTH-1)] == '\n')? 0: 1;
//if(blockIdx.x == 1) debug_tree[treeIdx-(WORDLENGTH-1)] = tree[treeIdx-(WORDLENGTH-1)];
        }
        else
            tree[treeIdx-(WORDLENGTH-1)] = 0;
    }
    __syncthreads();

    //debug
    //if(threadIdx.x < (WORDLENGTH-1) && threadIdx.x != 0) pos[textIdx-(WORDLENGTH-1)] = tree[treeIdx-(WORDLENGTH-1)]; 
    //pos[textIdx] = tree[treeIdx];

    //build the upper tree
    int parent, leftChild, rightChild;
    for(int p = N/2-1; p > 0; p = (p-1)/2 ) { //not including root
//int p = N/2-1;
        parent = threadIdx.x+p;
        leftChild = parent*2+1;
        rightChild = parent*2+2;
        if(threadIdx.x <= p) {
            tree[parent] = ( (tree[leftChild] & tree[rightChild]) == 0)? 0: tree[leftChild]+tree[rightChild];
//pos[ threadIdx.x + blockIdx.x*blockDim.x] = tree[parent];
//if(blockIdx.x == 1) debug_tree[parent] = tree[parent];
        }
        __syncthreads();
    }

    //root
    if((tree[1] & tree[2]) == 0) tree[0] = 0;
    else tree[0] = tree[1] + tree[2];
//if(blockIdx.x == 1) debug_tree[0] = tree[0];
   // printf("Tree[ %d ] = %d \n", treeIdx, tree[treeIdx]);

    //count position
    bool lastStep = 1;// 1 from right, 0 from child
    textIdx = threadIdx.x + blockIdx.x*blockDim.x;
    treeIdx = threadIdx.x+(N-1)+(WORDLENGTH-1);
    int idx = treeIdx;
    //count up
    while(idx >= 0 && tree[idx] != 0) {
        if(lastStep) pos[textIdx] += tree[idx];//from right
        else pos[textIdx] += (tree[idx]/2);//from child
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
    if( ( ( (idx+1) & -(idx+1) ) == (idx+1) && tree[idx] != 0) || idx >= N-1) return;//if the node is at left most side and is not zero, or it is the zero bottom node
    if(lastStep) idx = idx*2+2;//from right, next is right child
    else idx = idx*2+1;//from child
    while( idx > 0 && idx < N-1 && tree[idx] == 0) {
        idx = idx*2+2;
    }
    pos[textIdx]+=tree[idx];

}

void CountPosition(const char *text, int *pos, int text_size)
//void CountPosition(const char *text, int *pos, int* debug_tree, int text_size)
{
    //BuildTree_CountPosition<<<(text_size+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE>>>(text, pos);
    //BuildTree_CountPosition<<<(text_size+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE>>>(text, pos, debug_tree);
    CountPosition_slow<<<(text_size+1023)/1024, 1024>>>(text, pos);
}

struct isOne {
    __host__ __device__
    bool operator()(const int x) {
        return (x == 1);
    }
};

int ExtractHead(const int *pos, int *head, int text_size)
{
	int *buffer;
	int nhead;
	cudaMalloc(&buffer, sizeof(int)*text_size*2); // this is enough
	thrust::device_ptr<const int> pos_d(pos);
	thrust::device_ptr<int> head_d(head), flag_d(buffer), cumsum_d(buffer+text_size);

	// TODO
    thrust::sequence(flag_d, cumsum_d);
    nhead = thrust::copy_if(flag_d, cumsum_d, pos_d, head_d, isOne()) - head_d;
//do not touch this
	cudaFree(buffer);
	return nhead;
}

__global__ void Part3_transform(char* text, char* temp_text, int* pos) {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(pos[idx] != 0)  {
        if(pos[idx]%4 == 1 || pos[idx]%4 == 2) {
            if(pos[idx+2] != 0) temp_text[idx] = text[idx+2];
            else temp_text[idx] = '-';
        }
        else {
            temp_text[idx] = text[idx-2];
        }
    }
}

__global__ void Part3_assign(char* text, char* temp_text) {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    text[idx] = temp_text[idx];

//if(idx >= 300 && idx < 400) printf("text_gpu[%d] = %c\n", idx, text[idx]);
}

void Part3(char *text, int *pos, int *head, int text_size, int n_head)
{
//Characters of the 1st & the 3rd position swap; the 2nd & the 4th swap; 5th & 7th swap, etc. 
//For the last charaters (at most 2 characters) that don't have a partner to swap, fill them with '-'.

    char *temp_text;
    cudaMalloc(&temp_text, sizeof(char)*text_size);
    Part3_transform<<<(text_size+1023)/1024, 1024>>>(text, temp_text, pos);     
    Part3_assign<<<(text_size+1023)/1024, 1024>>>(text, temp_text);
    cudaFree(temp_text);    

}
