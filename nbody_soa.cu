#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define BLOCK 256

// forces -> velocities. Positions/mass now come in as separate arrays.
__global__ void computeForces(const float* x, const float* y, const float* z,
                              const float* mass, float* vx, float* vy, float* vz,
                              int n, float dt, float soft) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float px = (i < n) ? x[i] : 0.f;
    float py = (i < n) ? y[i] : 0.f;
    float pz = (i < n) ? z[i] : 0.f;
    float ax = 0.f, ay = 0.f, az = 0.f;

    __shared__ float sx[BLOCK], sy[BLOCK], sz[BLOCK], sm[BLOCK];

    for (int tile = 0; tile < n; tile += BLOCK) {
        int idx = tile + threadIdx.x;
        // coalesced: 32 neighboring threads read 32 neighboring x's in one go
        sx[threadIdx.x] = (idx < n) ? x[idx] : 0.f;
        sy[threadIdx.x] = (idx < n) ? y[idx] : 0.f;
        sz[threadIdx.x] = (idx < n) ? z[idx] : 0.f;
        sm[threadIdx.x] = (idx < n) ? mass[idx] : 0.f;
        __syncthreads();

        for (int j = 0; j < BLOCK; j++) {
            float dx = sx[j] - px;
            float dy = sy[j] - py;
            float dz = sz[j] - pz;
            float d2 = dx*dx + dy*dy + dz*dz + soft*soft;
            float inv = rsqrtf(d2);
            float s = sm[j] * inv * inv * inv;
            ax += dx*s; ay += dy*s; az += dz*s;
        }
        __syncthreads();
    }

    if (i < n) { vx[i] += dt*ax; vy[i] += dt*ay; vz[i] += dt*az; }
}

__global__ void integrate(float* x, float* y, float* z,
                          const float* vx, const float* vy, const float* vz,
                          int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += dt*vx[i]; y[i] += dt*vy[i]; z[i] += dt*vz[i];
}

int main() {
    int n = 50000, steps = 100;
    float dt = 0.01f, soft = 0.1f;
    size_t sz = (size_t)n * sizeof(float);

    // host arrays (SoA: one array per field)
    float *hx=(float*)malloc(sz), *hy=(float*)malloc(sz), *hz=(float*)malloc(sz);
    float *hm=(float*)malloc(sz), *hvx=(float*)malloc(sz), *hvy=(float*)malloc(sz), *hvz=(float*)malloc(sz);
    for (int i = 0; i < n; i++) {
        hx[i] = rand()/(float)RAND_MAX*2-1;
        hy[i] = rand()/(float)RAND_MAX*2-1;
        hz[i] = rand()/(float)RAND_MAX*2-1;
        hvx[i] = hvy[i] = hvz[i] = 0.f;
        hm[i] = 1.f;
    }

    // device arrays
    float *x,*y,*z,*m,*vx,*vy,*vz;
    cudaMalloc(&x,sz); cudaMalloc(&y,sz); cudaMalloc(&z,sz); cudaMalloc(&m,sz);
    cudaMalloc(&vx,sz); cudaMalloc(&vy,sz); cudaMalloc(&vz,sz);
    cudaMemcpy(x,hx,sz,cudaMemcpyHostToDevice);
    cudaMemcpy(y,hy,sz,cudaMemcpyHostToDevice);
    cudaMemcpy(z,hz,sz,cudaMemcpyHostToDevice);
    cudaMemcpy(m,hm,sz,cudaMemcpyHostToDevice);
    cudaMemcpy(vx,hvx,sz,cudaMemcpyHostToDevice);
    cudaMemcpy(vy,hvy,sz,cudaMemcpyHostToDevice);
    cudaMemcpy(vz,hvz,sz,cudaMemcpyHostToDevice);

    int threads = BLOCK, blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int s = 0; s < steps; s++) {
        computeForces<<<blocks,threads>>>(x,y,z,m,vx,vy,vz,n,dt,soft);
        integrate<<<blocks,threads>>>(x,y,z,vx,vy,vz,n,dt);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0; cudaEventElapsedTime(&ms, start, stop);

    printf("SoA GPU: %d bodies, %d steps -> %.2f ms\n", n, steps, ms);

    cudaFree(x);cudaFree(y);cudaFree(z);cudaFree(m);cudaFree(vx);cudaFree(vy);cudaFree(vz);
    free(hx);free(hy);free(hz);free(hm);free(hvx);free(hvy);free(hvz);
    return 0;
}
