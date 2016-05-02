#include "lab3.h"
#include <cstdio>
#include "SyncedMemory.h"

//static const float w = 1;
//static const int sampleConst = 2;

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void copyTargetToOutput(
    const float *target,
    const float *mask,
    float *output,
    const int wb, const int hb, const int wt, const int ht,
    const int oy, const int ox,
    const int sampleConst
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	const int yb = oy+yt, xb = ox+xt;
	const int curb = wb*yb+xb;
    if(0 <= yb and yb < hb and 0 <= xb and xb < wb and
       0 < yt and yt < (ht-sampleConst) and 0 < xt and xt < (wt-sampleConst) and
       mask[curt] > 127.0f ) {
        output[curb*3+0] = target[curt*3+0];
        output[curb*3+1] = target[curt*3+1];
        output[curb*3+2] = target[curt*3+2];
    } 
}

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	float *temp,
    const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox,
    const float w
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	const int yb = oy+yt, xb = ox+xt;
	const int curb = wb*yb+xb;
    if(0 <= yb and yb < hb and 0 <= xb and xb < wb and
       0 <= yt and yt < ht and 0 <= xt and xt < wt ) {
        temp[curt*3+0] = output[curb*3+0];
        temp[curt*3+1] = output[curb*3+1];
        temp[curt*3+2] = output[curb*3+2];
    }
    if (yt < (ht-1) and yt > 0 and xt < (wt-1) and xt > 0 and
        mask[curt] > 127.0f and 
        0 < yb and yb < (hb-1) and 0 < xb and xb < (wb-1) ) {
	    float gradient[3];
        float tmp[3];
	    gradient[0] = 4*target[curt*3+0]-(target[(curt-wt)*3+0]+target[(curt+wt)*3+0]+target[(curt-1)*3+0]+target[(curt+1)*3+0]);
	    gradient[1] = 4*target[curt*3+1]-(target[(curt-wt)*3+1]+target[(curt+wt)*3+1]+target[(curt-1)*3+1]+target[(curt+1)*3+1]);
	    gradient[2] = 4*target[curt*3+2]-(target[(curt-wt)*3+2]+target[(curt+wt)*3+2]+target[(curt-1)*3+2]+target[(curt+1)*3+2]);
	    tmp[0] = 0.25*(gradient[0]+output[(curb-wb)*3+0]+output[(curb+wb)*3+0]+output[(curb-1)*3+0]+output[(curb+1)*3+0]);
	    tmp[1] = 0.25*(gradient[1]+output[(curb-wb)*3+1]+output[(curb+wb)*3+1]+output[(curb-1)*3+1]+output[(curb+1)*3+1]);
	    tmp[2] = 0.25*(gradient[2]+output[(curb-wb)*3+2]+output[(curb+wb)*3+2]+output[(curb-1)*3+2]+output[(curb+1)*3+2]);

        temp[curt*3+0] = w*tmp[0]+(1-w)*output[curb*3+0];
        temp[curt*3+1] = w*tmp[1]+(1-w)*output[curb*3+1];
        temp[curt*3+2] = w*tmp[2]+(1-w)*output[curb*3+2];
	}
}
__global__ void assignBack(
    float *output,
    float *temp,
    const int wb, const int hb, const int wt, const int ht,
    const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	const int yb = oy+yt, xb = ox+xt;
	const int curb = wb*yb+xb;
    if(0 <= yb and yb < hb and 0 <= xb and xb < wb and
       0 <= yt and yt < ht and 0 <= xt and xt < wt ) {
        output[curb*3+0] = temp[curt*3+0];
        output[curb*3+1] = temp[curt*3+1];
        output[curb*3+2] = temp[curt*3+2];
    }
}
__global__ void downsampleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox,
    const int sampleConst
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	const int yb = oy+yt, xb = ox+xt;
	const int curb = wb*yb+xb;
	float temp[3];
    if(0 <= yb and yb < hb and 0 <= xb and xb < wb and
       !(yt % sampleConst) and !(xt % sampleConst) ) {
        temp[0] = output[curb*3+0];
        temp[1] = output[curb*3+1];
        temp[2] = output[curb*3+2];
    }
    __syncthreads();
    if (yt < (ht-sampleConst) and yt > 0 and xt < (wt-sampleConst) and xt > 0 and
        mask[curt] > 127.0f and /*mask[curt+1] > 127.0f and mask[curt+wt] > 127.0f and mask[curt+wt+1] > 127.0f and*/
        0 < yb and yb < (hb-sampleConst) and 0 < xb and xb < (wb-sampleConst) and
        !(yt % sampleConst) and !(xt % sampleConst) ) {
	    float gradient[3];

	    gradient[0] = 4*target[curt*3+0]-(target[(curt-sampleConst*wt)*3+0]+target[(curt+sampleConst*wt)*3+0]+target[(curt-sampleConst)*3+0]+target[(curt+sampleConst)*3+0]);
	    gradient[1] = 4*target[curt*3+1]-(target[(curt-sampleConst*wt)*3+1]+target[(curt+sampleConst*wt)*3+1]+target[(curt-sampleConst)*3+1]+target[(curt+sampleConst)*3+1]);
	    gradient[2] = 4*target[curt*3+2]-(target[(curt-sampleConst*wt)*3+2]+target[(curt+sampleConst*wt)*3+2]+target[(curt-sampleConst)*3+2]+target[(curt+sampleConst)*3+2]);
	    temp[0] = 0.25*(gradient[0]+output[(curb-sampleConst*wb)*3+0]+output[(curb+sampleConst*wb)*3+0]+output[(curb-sampleConst)*3+0]+output[(curb+sampleConst)*3+0]);
	    temp[1] = 0.25*(gradient[1]+output[(curb-sampleConst*wb)*3+1]+output[(curb+sampleConst*wb)*3+1]+output[(curb-sampleConst)*3+1]+output[(curb+sampleConst)*3+1]);
	    temp[2] = 0.25*(gradient[2]+output[(curb-sampleConst*wb)*3+2]+output[(curb+sampleConst*wb)*3+2]+output[(curb-sampleConst)*3+2]+output[(curb+sampleConst)*3+2]);
/*      
	    gradient[0] = 4*target[curt*3+0]-(target[(curt-wt)*3+0]+target[(curt+wt)*3+0]+target[(curt-1)*3+0]+target[(curt+1)*3+0]);
	    gradient[1] = 4*target[curt*3+1]-(target[(curt-wt)*3+1]+target[(curt+wt)*3+1]+target[(curt-1)*3+1]+target[(curt+1)*3+1]);
	    gradient[2] = 4*target[curt*3+2]-(target[(curt-wt)*3+2]+target[(curt+wt)*3+2]+target[(curt-1)*3+2]+target[(curt+1)*3+2]);
	    temp[0] = 0.25*(gradient[0]+output[(curb-wb)*3+0]+output[(curb+wb)*3+0]+output[(curb-1)*3+0]+output[(curb+1)*3+0]);
	    temp[1] = 0.25*(gradient[1]+output[(curb-wb)*3+1]+output[(curb+wb)*3+1]+output[(curb-1)*3+1]+output[(curb+1)*3+1]);
	    temp[2] = 0.25*(gradient[2]+output[(curb-wb)*3+2]+output[(curb+wb)*3+2]+output[(curb-1)*3+2]+output[(curb+1)*3+2]);
*/    }
    if(0 <= yb and yb < hb and 0 <= xb and xb < wb and
       !(yt % sampleConst) and !(xt % sampleConst) ) {
        output[curb*3+0] = temp[0];
        output[curb*3+1] = temp[1];
        output[curb*3+2] = temp[2];
    }
    __syncthreads();

}
__global__ void upsample(
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox,
    const int sampleConst
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	const int yb = oy+yt, xb = ox+xt;
	const int curb = wb*yb+xb;
    if(0 <= yb and yb < hb and 0 <= xb and xb < wb and
       0 <= yt and yt < ht and 0 <= xt and xt < wt and
       ( (yt % sampleConst) or (xt % sampleConst) ) and
       mask[curt] > 127.0f)
    {
        const int sam_xt = xt & ~(sampleConst-1);
        const int sam_yt = yt & ~(sampleConst-1);
        const int sam_curb = wb*(oy+sam_yt)+(ox+sam_xt);
        output[curb*3+0] = output[sam_curb*3+0];
        output[curb*3+1] = output[sam_curb*3+1];
        output[curb*3+2] = output[sam_curb*3+2];
    }
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
	const int SIZET = wt*ht*3;
	MemoryBuffer<float> temp_o(SIZET);
	auto temp_s = temp_o.CreateSync(SIZET);
    float *temp = temp_s.get_gpu_wo();
    float w = 1.0f;
//    int sampleConst = 4;

	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);

    copyTargetToOutput<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
        target, mask, output,
        wb, hb, wt, ht, oy, ox, 8
    );

    for(int i = 0; i != 400; ++i)
    downsampleClone<<<dim3(CeilDiv(wt, 32), CeilDiv(ht, 16)), dim3(32, 16)>>>(
		background, target, mask, output,
		wb, hb, wt, ht, oy, ox, 8
    );

    upsample<<<dim3(CeilDiv(wt, 32), CeilDiv(ht, 16)), dim3(32, 16)>>>(
        mask, output,
        wb, hb, wt, ht, oy, ox, 8
    );

    for(int i = 0; i != 200; ++i)
    downsampleClone<<<dim3(CeilDiv(wt, 32), CeilDiv(ht, 16)), dim3(32, 16)>>>(
		background, target, mask, output,
		wb, hb, wt, ht, oy, ox, 4
    );

    upsample<<<dim3(CeilDiv(wt, 32), CeilDiv(ht, 16)), dim3(32, 16)>>>(
        mask, output,
        wb, hb, wt, ht, oy, ox, 4
    );


    for (int i = 0; i != 100; ++i)
    downsampleClone<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
        background, target, mask, output,
        wb, hb, wt, ht, oy, ox, 2
    );

    upsample<<<dim3(CeilDiv(wt, 32), CeilDiv(ht, 16)), dim3(32, 16)>>>(
        mask, output,
        wb, hb, wt, ht, oy, ox, 2
    );

	for(int i = 0; i != 50; ++i) {
        //w = (i < 50)? 1.05f: 1;
	    SimpleClone<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
		    background, target, mask, output, temp,
		    wb, hb, wt, ht, oy, ox, w
        );
        assignBack<<<dim3(CeilDiv(wt, 32), CeilDiv(ht, 16)), dim3(32, 16)>>>(
            output, temp,
            wb, hb, wt, ht, oy, ox
        );
    }

}
