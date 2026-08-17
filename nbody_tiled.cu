#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define BLOCK 256

struct Body { float x, y, z, vx, vy, vz, mass; };

__global__ void computeForces(Body* b, int n, float dt, float soft) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float px = (i < n) ? b[i].x : 0.f;   // my particle's position
    float py = (i < n) ? b[i].y : 0.f;
    float pz = (i < n) ? b[i].z : 0.f;
    float ax = 0.f, ay = 0.f, az = 0.f;

    __shared__ float sx[BLOCK], sy[BLOCK], sz[BLOCK], sm[BLOCK]; // the whiteboard

    // walk the 50k particles one tile (256) at a time
    for (int tile = 0; tile < n; tile += BLOCK) {
        int idx = tile + threadIdx.x;
        // each thread loads ONE particle into shared memory
        if (idx < n) {
            sx[threadIdx.x] = b[idx].x;
            sy[threadIdx.x] = b[idx].y;
            sz[threadIdx.x] = b[idx].z;
            sm[threadIdx.x] = b[idx].mass;
        } else {
            sm[threadIdx.x] = 0.f; // padding particle contributes nothing
        }
        __syncthreads(); // wait until the whole tile is loaded

        // now read all 256 from fast shared memory
        for (int j = 0; j < BLOCK; j++) {
            float dx = sx[j] - px;
            float dy = sy[j] - py;
            float dz = sz[j] - pz;
            float d2 = dx*dx + dy*dy + dz*dz + soft*soft;
            float inv = rsqrtf(d2);
            float s = sm[j] * inv * inv * inv;
            ax += dx*s; ay += dy*s; az += dz*s;
        }
        __syncthreads(); // wait before overwriting the whiteboard with next tile
    }

    if (i < n) { b[i].vx += dt*ax; b[i].vy += dt*ay; b[i].vz += dt*az; }
}

__global__ void integrate(Body* b, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    b[i].x += dt*b[i].vx; b[i].y += dt*b[i].vy; b[i].z += dt*b[i].vz;
}

int main() {
    int n = 50000, steps = 100;
    float dt = 0.01f, soft = 0.1f;
    size_t bytes = (size_t)n * sizeof(Body);

    Body* h = (Body*)malloc(bytes);
    for (int i = 0; i < n; i++) {
        h[i].x = rand()/(float)RAND_MAX*2-1;
        h[i].y = rand()/(float)RAND_MAX*2-1;
        h[i].z = rand()/(float)RAND_MAX*2-1;
        h[i].vx = h[i].vy = h[i].vz = 0;
        h[i].mass = 1.f;
    }

    Body* d;
    cudaMalloc(&d, bytes);
    cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);

    int threads = BLOCK;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int s = 0; s < steps; s++) {
        computeForces<<<blocks, threads>>>(d, n, dt, soft);
        integrate<<<blocks, threads>>>(d, n, dt);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0; cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost);
    printf("TILED GPU: %d bodies, %d steps -> %.2f ms\n", n, steps, ms);

    cudaFree(d); free(h);
    return 0;
}
