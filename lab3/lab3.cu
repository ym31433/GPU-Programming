#include "lab3.h"
#include <cstdio>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	const int yb = oy+yt, xb = ox+xt;
	const int curb = wb*yb+xb;
	float temp[3];
    if(0 <= yb and yb < hb and 0 <= xb and xb < wb) {
        temp[0] = output[curb*3+0];
        temp[1] = output[curb*3+1];
        temp[2] = output[curb*3+2];
    }
	if (yt < (ht-1) and yt > 0 and xt < (wt-1) and xt > 0 and mask[curt] > 127.0f) {
		float gradient[3];
		gradient[0] = 4*target[curt*3+0]-(target[(curt-wt)*3+0]+target[(curt+wt)*3+0]+target[(curt-1)*3+0]+target[(curt+1)*3+0]);
		gradient[1] = 4*target[curt*3+1]-(target[(curt-wt)*3+1]+target[(curt+wt)*3+1]+target[(curt-1)*3+1]+target[(curt+1)*3+1]);
		gradient[2] = 4*target[curt*3+2]-(target[(curt-wt)*3+2]+target[(curt+wt)*3+2]+target[(curt-1)*3+2]+target[(curt+1)*3+2]);
		if (0 < yb and yb < (hb-1) and 0 < xb and xb < (wb-1)) {
			temp[0] = 0.25*(gradient[0]+output[(curb-wb)*3+0]+output[(curb+wb)*3+0]+output[(curb-1)*3+0]+output[(curb+1)*3+0]);
			temp[1] = 0.25*(gradient[1]+output[(curb-wb)*3+1]+output[(curb+wb)*3+1]+output[(curb-1)*3+1]+output[(curb+1)*3+1]);
			temp[2] = 0.25*(gradient[2]+output[(curb-wb)*3+2]+output[(curb+wb)*3+2]+output[(curb-1)*3+2]+output[(curb+1)*3+2]);
		}
	}
    __syncthreads();
    if(0 <= yb and yb < hb and 0 <= xb and xb < wb) {
        output[curb*3+0] = temp[0];
        output[curb*3+1] = temp[1];
        output[curb*3+2] = temp[2];
    }
    __syncthreads();
}

void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	for(int i = 0; i != 20000; ++i)
	SimpleClone<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
		background, target, mask, output,
		wb, hb, wt, ht, oy, ox
	);
}
