#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK 256

__global__ void computeForces(const float* x, const float* y, const float* z,
                              const float* mass, float* vx, float* vy, float* vz,
                              int n, float dt, float soft) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float px=(i<n)?x[i]:0.f, py=(i<n)?y[i]:0.f, pz=(i<n)?z[i]:0.f;
    float ax=0,ay=0,az=0;
    __shared__ float sx[BLOCK],sy[BLOCK],sz[BLOCK],sm[BLOCK];
    for (int tile=0; tile<n; tile+=BLOCK){
        int idx=tile+threadIdx.x;
        sx[threadIdx.x]=(idx<n)?x[idx]:0.f; sy[threadIdx.x]=(idx<n)?y[idx]:0.f;
        sz[threadIdx.x]=(idx<n)?z[idx]:0.f; sm[threadIdx.x]=(idx<n)?mass[idx]:0.f;
        __syncthreads();
        for (int j=0;j<BLOCK;j++){
            float dx=sx[j]-px, dy=sy[j]-py, dz=sz[j]-pz;
            float d2=dx*dx+dy*dy+dz*dz+soft*soft;
            float inv=rsqrtf(d2), s=sm[j]*inv*inv*inv;
            ax+=dx*s; ay+=dy*s; az+=dz*s;
        }
        __syncthreads();
    }
    if(i<n){ vx[i]+=dt*ax; vy[i]+=dt*ay; vz[i]+=dt*az; }
}
__global__ void integrate(float*x,float*y,float*z,const float*vx,const float*vy,const float*vz,int n,float dt){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    x[i]+=dt*vx[i]; y[i]+=dt*vy[i]; z[i]+=dt*vz[i];
}

int main(){
    int n=20000, steps=400, every=4;
    float dt=0.001f, soft=0.15f;           // tiny step + solid softening = stable
    float Mcenter=50.f;                     // modest core, not a monster
    size_t sz=(size_t)n*sizeof(float);
    float *hx=(float*)malloc(sz),*hy=(float*)malloc(sz),*hz=(float*)malloc(sz);
    float *hm=(float*)malloc(sz),*hvx=(float*)malloc(sz),*hvy=(float*)malloc(sz),*hvz=(float*)malloc(sz);

    hx[0]=0;hy[0]=0;hz[0]=0;hvx[0]=hvy[0]=hvz[0]=0;hm[0]=Mcenter;

    for(int i=1;i<n;i++){
        float r = 0.3f + 1.2f*sqrtf(rand()/(float)RAND_MAX);
        float a = rand()/(float)RAND_MAX*6.2831853f;
        hx[i]=r*cosf(a); hy[i]=r*sinf(a); hz[i]=(rand()/(float)RAND_MAX-0.5f)*0.02f;
        // circular-orbit speed using SOFTENED gravity, scaled to 0.9 so it settles inward slightly
        float v = 0.9f * sqrtf(Mcenter / sqrtf(r*r + soft*soft));
        hvx[i]=-v*sinf(a); hvy[i]=v*cosf(a); hvz[i]=0;
        hm[i]=1.f;
    }

    float *x,*y,*z,*m,*vx,*vy,*vz;
    cudaMalloc(&x,sz);cudaMalloc(&y,sz);cudaMalloc(&z,sz);cudaMalloc(&m,sz);
    cudaMalloc(&vx,sz);cudaMalloc(&vy,sz);cudaMalloc(&vz,sz);
    cudaMemcpy(x,hx,sz,cudaMemcpyHostToDevice); cudaMemcpy(y,hy,sz,cudaMemcpyHostToDevice);
    cudaMemcpy(z,hz,sz,cudaMemcpyHostToDevice); cudaMemcpy(m,hm,sz,cudaMemcpyHostToDevice);
    cudaMemcpy(vx,hvx,sz,cudaMemcpyHostToDevice); cudaMemcpy(vy,hvy,sz,cudaMemcpyHostToDevice);
    cudaMemcpy(vz,hvz,sz,cudaMemcpyHostToDevice);

    int threads=BLOCK, blocks=(n+threads-1)/threads;
    FILE* f=fopen("frames.bin","wb"); fwrite(&n,sizeof(int),1,f);

    for(int s=0;s<steps;s++){
        computeForces<<<blocks,threads>>>(x,y,z,m,vx,vy,vz,n,dt,soft);
        integrate<<<blocks,threads>>>(x,y,z,vx,vy,vz,n,dt);
        if(s%every==0){
            cudaMemcpy(hx,x,sz,cudaMemcpyDeviceToHost);
            cudaMemcpy(hy,y,sz,cudaMemcpyDeviceToHost);
            fwrite(hx,sizeof(float),n,f); fwrite(hy,sizeof(float),n,f);
        }
    }
    fclose(f); printf("done\n"); return 0;
}
