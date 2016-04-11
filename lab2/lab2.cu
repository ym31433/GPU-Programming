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
static const unsigned FACTORT = 5;
static const double g = 9.8;
static const unsigned rBall = 20;
static const unsigned aEllipse = 20;
static const unsigned bEllipse = 8;
static const double lightX = 10;
static const double lightY = 0;
static const unsigned n_balls = 7; //number of balls

struct Lab2VideoGenerator::Impl {
	int t = 0;
    int* t_start = new int[n_balls];
    double* vx = new double[n_balls];
    double* vy = new double[n_balls];
    double* x0 = new double[n_balls];
    double* y0 = new double[n_balls];
};

Lab2VideoGenerator::Lab2VideoGenerator(): impl(new Impl) {
    srand(0);
    for(int i = 0; i != n_balls; ++i) {
        impl->t_start[i] = 0;
        impl->vx[i] = (i&1)? -10: 10;
        impl->vy[i] = 10;
    }
    impl->x0[0] = 20;//rand()%W;
    impl->y0[0] = 20;//rand()/W;
    impl->x0[1] = 120;//rand()%W;
    impl->y0[1] = 25;//rand()/W;
    impl->x0[2] = 220;//rand()%W;
    impl->y0[2] = 30;//rand()/W;
    impl->x0[3] = 320;//rand()%W;
    impl->y0[3] = 50;//rand()/W;
    impl->x0[4] = 420;//rand()%W;
    impl->y0[4] = 40;//rand()/W;
    impl->x0[5] = 180;//rand()%W;
    impl->y0[5] = 100;//rand()/W;
    impl->x0[6] = 600;//rand()%W;
    impl->y0[6] = 40;//rand()/W;

}

Lab2VideoGenerator::~Lab2VideoGenerator() {
    delete [] impl->t_start;
    delete [] impl->vx;
    delete [] impl->vy;
    delete [] impl->x0;
    delete [] impl->y0;
}


void Lab2VideoGenerator::get_info(Lab2VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
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
__device__ bool isInEllipse(const int& x, const int& y, const double& x_hat, const double& y_hat) {
    double factor = y_hat/480;
    double a = factor*(double)aEllipse;
    double b = factor*(double)bEllipse;
    double shadowX = x_hat+(479-y_hat)*(x_hat-lightX)/(y_hat-lightY);
    double X = ( ((double)x) -shadowX)*( ((double)x) -shadowX);
    double Y = ( ((double)y) - 470 )*( ((double)y) - 470 );
    return X/(a*a)+Y/(b*b) <= 1;
}
__device__ void fillUV(uint8_t* yuv, const int& x, const int& y, const uint8_t& U, const uint8_t& V) {
    if(!(x & 1) && !(y & 1)) {
        yuv[SIZE+(y/2)*(W/2)+(x/2)] = U;
        yuv[SIZE*5/4+(y/2)*(W/2)+(x/2)] = V;
    }
}

__global__ void frame_generate(uint8_t* yuv, const uint8_t* t, const double* coor) {
    int idx = threadIdx.x+blockIdx.x*blockDim.x;
    int y = idx/W;
    int x = idx%W;
//printf("x: %d, y: %d\n", x, y);
//sine
/*
    int sine1 = sine(x, t, 60, 1);
    int sine2 = sine(x, t, 40, 0);
    if(isSine(y, sine1+sine2+240)) yuv[idx] = 0;
    else if(isSine(y, sine1+240)) yuv[idx] = 100;
    else if(isSine(y, sine2+240)) yuv[idx] = 170;
    else yuv[idx] = 255;
*/
    yuv[idx] = 255;
    fillUV(yuv, x, y, 128, 128);

    //shadow
    for(int i = 0; i != n_balls; ++i) {
        if(isInEllipse(x, y, coor[i*2+1], coor[i*2])) {
            yuv[idx] = 128;
            fillUV(yuv, x, y, 128, 128);
        }
    }
    //ball
    if(isInBall(x, y, (int)coor[1], (int)coor[0], rBall)) {
        yuv[idx] = 76;
        fillUV(yuv, x, y, 84, 225);
    }
    if(isInBall(x, y, (int)coor[3], (int)coor[2], rBall)) {
        yuv[idx] = 204;
        fillUV(yuv, x, y, 29, 153);
    }
    if(isInBall(x, y, (int)coor[5], (int)coor[4], rBall)) {
        yuv[idx] = 29;
        fillUV(yuv, x, y, 255, 107);
    }
    if(isInBall(x, y, (int)coor[7], (int)coor[6], rBall)) {
        yuv[idx] = 202;
        fillUV(yuv, x, y, 126, 54);
    }
    if(isInBall(x, y, (int)coor[9], (int)coor[8], rBall)) {
        yuv[idx] = 52;
        fillUV(yuv, x, y, 211, 161);
    }
    if(isInBall(x, y, (int)coor[11], (int)coor[10], rBall)) {
        yuv[idx] = 216;
        fillUV(yuv, x, y, 118, 155);
    }
    if(isInBall(x, y, (int)coor[13], (int)coor[12], rBall)) {
        yuv[idx] = 161;
        fillUV(yuv, x, y, 104, 155);
    }
/*//light
    else if(isInBall(x, y, (int)lightX, (int)lightY, 10)) {
        yuv[idx] = 204;
        fillUV(yuv, x, y, 29, 153);
    }
*/
}

void Lab2VideoGenerator::Generate(uint8_t *yuv) {
    MemoryBuffer<uint8_t> t(1);
    auto t_sync = t.CreateSync(1);
    *(t_sync.get_cpu_wo()) = impl->t;

    MemoryBuffer<double> coordinates(n_balls*2);
    auto c_sync = coordinates.CreateSync(n_balls*2);
    double* c_ptr = c_sync.get_cpu_rw();
//cout << "time = " << time << endl;
    for(int i = 0; i != n_balls; ++i) {
        double time = (double)(impl->t - impl->t_start[i])/FACTORT;
        c_ptr[i*2] = impl->y0[i] + (impl->vy[i])*time + g*time*time/2;
        c_ptr[i*2+1] = impl->x0[i] + impl->vx[i]*time;
    }
//cout << "(x, y) = (" << c_ptr[1] << ", " << c_ptr[0] << ")" << endl;

    frame_generate<<<(SIZE+1023)/1024, 1024>>>(yuv, t_sync.get_gpu_ro(), c_sync.get_gpu_ro());

//ball collision
    for(int i = 0; i != n_balls-1; ++i) {
        for(int j = i+1; j != n_balls; ++j) {
            double deltaX = c_ptr[i*2+1] - c_ptr[j*2+1];
            double deltaY = c_ptr[i*2] - c_ptr[j*2];
            double lengthBall = deltaX*deltaX+deltaY*deltaY;
            if(lengthBall <= 4*(double)rBall*(double)rBall) {
            //if(lengthBall <= 4*(double)rBall*(double)rBall && (impl->t - impl->t_start[i]) >= 13 && (impl->t - impl->t_start[j]) >= 13) {
                double centerX = (c_ptr[i*2+1]+c_ptr[j*2+1])/2;
                double centerY = (c_ptr[i*2]+c_ptr[j*2])/2;
                c_ptr[i*2+1] = centerX+deltaX*(rBall+1)/sqrt(lengthBall);
                c_ptr[i*2] = centerY+deltaY*(rBall+1)/sqrt(lengthBall);
                c_ptr[j*2+1] = centerX-deltaX*(rBall+1)/sqrt(lengthBall);
                c_ptr[j*2] = centerY-deltaY*(rBall+1)/sqrt(lengthBall);
                deltaX = c_ptr[i*2+1] - c_ptr[j*2+1];
                deltaY = c_ptr[i*2] - c_ptr[j*2];
                lengthBall = deltaX*deltaX+deltaY*deltaY;

                double lengthI = (impl->vx[i])*(impl->vx[i])+(impl->vy[i])*(impl->vy[i]);
                double cosI = ( (impl->vx[i]*deltaX+impl->vy[i]*deltaY)/(sqrt(lengthBall)*sqrt(lengthI)) );
                double vnxI = deltaX*sqrt(lengthI)*cosI/sqrt(lengthBall);
                double vnyI = deltaY*sqrt(lengthI)*cosI/sqrt(lengthBall);
                double vtxI = impl->vx[i] - vnxI;
                double vtyI = impl->vy[i] - vnyI;
//if(vnxI*vtxI+vnyI*vtyI != 0) cout << "error: dot product is " <<vnxI*vtxI+vnyI*vtyI << endl; 

                double lengthJ = (impl->vx[j])*(impl->vx[j])+(impl->vy[j])*(impl->vy[j]);
                double cosJ = ( (impl->vx[j]*deltaX+impl->vy[j]*deltaY)/(sqrt(lengthBall)*sqrt(lengthJ)) );
                double vnxJ = deltaX*sqrt(lengthJ)*cosJ/sqrt(lengthBall);
                double vnyJ = deltaY*sqrt(lengthJ)*cosJ/sqrt(lengthBall);
                double vtxJ = impl->vx[j] - vnxJ;
                double vtyJ = impl->vy[j] - vnyJ;
        
                impl->vx[i] = 1*(vnxJ+vtxI);
                impl->vy[i] = 1*(vnyJ+vtyI);
                impl->vx[j] = 1*(vnxI+vtxJ);
                impl->vy[j] = 1*(vnyI+vtyJ);

                //lengthI = (impl->vx[i])*(impl->vx[i])+(impl->vy[i])*(impl->vy[i]);
                //lengthJ = (impl->vx[j])*(impl->vx[j])+(impl->vy[j])*(impl->vy[j]);
                impl->t_start[i] = impl->t;
                impl->t_start[j] = impl->t;
                impl->x0[i] = c_ptr[i*2+1];//+impl->vx[i]*30/(FACTORT*sqrt(lengthI));
                impl->y0[i] = c_ptr[i*2];//+impl->vy[i]*30/(FACTORT*sqrt(lengthI));
                impl->x0[j] = c_ptr[j*2+1];//+impl->vx[j]*30/(FACTORT*sqrt(lengthJ));
                impl->y0[j] = c_ptr[j*2];//+impl->vy[j]*30/(FACTORT*sqrt(lengthJ));
            }
        }
    }
    for(int i = 0; i != n_balls; ++i) {
//floor
        if(c_ptr[i*2] >= H-rBall-bEllipse/2 && (impl->t != impl->t_start[i])) {
            impl->vy[i] = -0.9*abs(impl->vy[i] + g*(double)(impl->t - impl->t_start[i])/FACTORT);
            impl->t_start[i] = impl->t;
            impl->x0[i] = c_ptr[i*2+1];
            impl->y0[i] = c_ptr[i*2];
        }
//wall
        else if((c_ptr[i*2+1] <= rBall) && (impl->t != impl->t_start[i])) {
        //else if((c_ptr[i*2+1] <= rBall || c_ptr[i*2+1] >= W-rBall) && impl->t - impl->t_start[i] >= 2) {
            impl->vx[i] = 0.9*abs(impl->vx[i]);
            impl->vy[i] = (impl->vy[i] + g*(double)(impl->t - impl->t_start[i])/FACTORT);
            impl->t_start[i] = impl->t;
            impl->x0[i] = c_ptr[i*2+1];
            impl->y0[i] = c_ptr[i*2];
        }
        else if((c_ptr[i*2+1] >= W-rBall) && (impl->t != impl->t_start[i])) {
        //else if((c_ptr[i*2+1] <= rBall || c_ptr[i*2+1] >= W-rBall) && impl->t - impl->t_start[i] >= 2) {
            impl->vx[i] = -0.9*abs(impl->vx[i]);
            impl->vy[i] = (impl->vy[i] + g*(double)(impl->t - impl->t_start[i])/FACTORT);
            impl->t_start[i] = impl->t;
            impl->x0[i] = c_ptr[i*2+1];
            impl->y0[i] = c_ptr[i*2];
        }
//ceiling
        else if(c_ptr[i*2] <= rBall && (impl->t != impl->t_start[i])) {
            impl->vy[i] = 0.9*abs(impl->vy[i] + g*(double)(impl->t - impl->t_start[i])/FACTORT);
            impl->t_start[i] = impl->t;
            impl->x0[i] = c_ptr[i*2+1];
            impl->y0[i] = c_ptr[i*2];
            
        }
    }

	++(impl->t);

}
