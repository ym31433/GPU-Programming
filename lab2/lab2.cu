#include "lab2.h"
#include <stdlib.h>
#include <math.h>
#include "SyncedMemory.h"
#include <iostream>
using namespace std;
static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 240;
static const unsigned SIZE = W*H;
static const unsigned FACTORX = 50;
static const unsigned FACTORT = 50;
static const double v = 10;
static const double g = 9.8;
//static const unsigned sps = 40;
//static const unsigned xn = W/sps;
//static const unsigned yn = H/sps;

struct Lab2VideoGenerator::Impl {
	int t = 0;
};

Lab2VideoGenerator::Lab2VideoGenerator(): impl(new Impl) {
/*
    U = new double*[yn+1];
    V = new double*[yn+1];
    for(int i = 0; i != yn+1; ++i) {
        U[i] = new double[xn+1];
        V[i] = new double[xn+1];
    }
*/
}

Lab2VideoGenerator::~Lab2VideoGenerator() {
/*
    for(int i = 0; i != yn+1; ++i) {
        delete [] U[i];
        delete [] V[i];
    }
    delete [] U;
    delete [] V;
*/
}

void Lab2VideoGenerator::get_info(Lab2VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
/*
    info.X = new double*[H];
    info.Y = new double*[H];
    for(int i = 0; i != W; ++i) {
        info.X[i] = new double[W];
        info.Y[i] = new double[W];
    }
    int w = xn/(W-1);
    int h = yn/(H-1);
    for(int i = 0; i != H; ++i) {
        for(int j = 0; j != W; ++j) {
            X[i][j] = w*j;
            Y[i][j] = h*i;
        }
    }
*/
};

__device__ int sine(const int& x, const uint8_t* t, const int& factorY, const bool& right) {
    if(right)
        return (int)(factorY*sin((float)x/FACTORX-(float)(*t)/FACTORT));
    else
        return (int)(factorY*sin((float)x/FACTORX-(float)(640-*t)/FACTORT));
//    printf("f is %d\n", f);
}
__device__ bool isSine(const int& y, const int& f) {
    return y == f || y == (f+1) || y == (f-1);
}
__device__ bool isInBall(const int& x, const int& y, const int& x_hat, const int& y_hat, const int& r) {
    return (x-x_hat)*(x-x_hat)+(y-y_hat)*(y-y_hat) <= r*r;
}

__global__ void frame_generate(uint8_t* yuv, const uint8_t* t) {
    int idx = threadIdx.x+blockIdx.x*blockDim.x;
    int y = idx/W;
    int x = idx%W;
//printf("x: %d, y: %d\n", x, y);
//sine

    int sine1 = sine(x, t, 60, 1);
    int sine2 = sine(x, t, 40, 0);
    if(isSine(y, sine1+sine2+240)) yuv[idx] = 0;
    else if(isSine(y, sine1+240)) yuv[idx] = 100;
    else if(isSine(y, sine2+240)) yuv[idx] = 170;
    else yuv[idx] = 255;

//ball
/*
    double time = (*t)/FACTORT;
    int x_hat = (int)(v*time);
    int y_hat = (int)(g*time*time/2);
    if(isInBall(x, y, x_hat, y_hat, 10)) yuv[idx] = 0;
    else yuv[idx] = 255;
*/
}

void Lab2VideoGenerator::Generate(uint8_t *yuv) {
//perlin noise
/*
    for(int i = 0; i != yn+1; ++i) {
        for(int j = 0; j != xn+1; ++j) {
            U[i][j] = rand()*2/RAND_MAX - 1;
            V[i][j] = rand()*2/RAND_MAX - 1;
        }
    }
    int xc, yc;
    double//contiune from here
    for(int i = 0; i != H; ++i) {
        for(int j = 0; j != W; ++j) {
            xc = (int)X[i][j];
            yc = (int)Y[i][j];
            if( !fmod(X[i][j],1) && !xc ) xc = xc-1;
            if( !fmod(Y[i][j],1) && !yc ) yc = yc-1;
        }
    }
*/
    MemoryBuffer<uint8_t> t(1);
    auto t_sync = t.CreateSync(1);
    *(t_sync.get_cpu_wo()) = impl->t;
    frame_generate<<<(SIZE+1023)/1024, 1024>>>(yuv, t_sync.get_gpu_ro());
//	cudaMemset(yuv, (impl->t)*255/NFRAME, W*H);
	cudaMemset(yuv+W*H, 128, W*H/2);
	++(impl->t);

}
