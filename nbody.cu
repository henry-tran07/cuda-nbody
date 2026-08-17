#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

struct Body { float x, y, z, vx, vy, vz, mass; };

// One thread = one particle. Sums gravity from all others, updates its velocity.
__global__ void computeForces(Body* b, int n, float dt, float soft) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // which particle am I
    if (i >= n) return;
    float ax = 0.f, ay = 0.f, az = 0.f;
    for (int j = 0; j < n; j++) {
        float dx = b[j].x - b[i].x;
        float dy = b[j].y - b[i].y;
        float dz = b[j].z - b[i].z;
        float d2 = dx*dx + dy*dy + dz*dz + soft*soft;  // soft avoids /0
        float inv = rsqrtf(d2);
        float s = b[j].mass * inv * inv * inv;
        ax += dx*s; ay += dy*s; az += dz*s;
    }
    b[i].vx += dt*ax; b[i].vy += dt*ay; b[i].vz += dt*az;
}

// Separate pass moves the particles (kept apart to avoid a data race).
__global__ void integrate(Body* b, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    b[i].x += dt*b[i].vx; b[i].y += dt*b[i].vy; b[i].z += dt*b[i].vz;
}

int main() {
    int n = 50000, steps = 100;          // MUST match your CPU run
    float dt = 0.01f, soft = 0.1f;
    size_t bytes = (size_t)n * sizeof(Body);

    Body* h = (Body*)malloc(bytes);      // particles on the CPU
    for (int i = 0; i < n; i++) {
        h[i].x = rand()/(float)RAND_MAX*2-1;
        h[i].y = rand()/(float)RAND_MAX*2-1;
        h[i].z = rand()/(float)RAND_MAX*2-1;
        h[i].vx = h[i].vy = h[i].vz = 0;
        h[i].mass = 1.f;
    }

    Body* d;                             // copy them to the GPU
    cudaMalloc(&d, bytes);
    cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;             // time the GPU work
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int s = 0; s < steps; s++) {
        computeForces<<<blocks, threads>>>(d, n, dt, soft);
        integrate<<<blocks, threads>>>(d, n, dt);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0; cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost);  // copy results back
    printf("GPU: %d bodies, %d steps -> %.2f ms\n", n, steps, ms);

    cudaFree(d); free(h);
    return 0;
}
