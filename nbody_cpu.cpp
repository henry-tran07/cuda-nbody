#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

struct Body { float x, y, z, vx, vy, vz, mass; };

int main() {
    int n = 50000, steps = 100;          // MUST match the GPU run
    float dt = 0.01f, soft = 0.1f;

    Body* b = (Body*)malloc(n * sizeof(Body));
    for (int i = 0; i < n; i++) {
        b[i].x = rand()/(float)RAND_MAX*2-1;
        b[i].y = rand()/(float)RAND_MAX*2-1;
        b[i].z = rand()/(float)RAND_MAX*2-1;
        b[i].vx = b[i].vy = b[i].vz = 0;
        b[i].mass = 1.f;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int s = 0; s < steps; s++) {
        // forces -> velocities (one particle at a time, the CPU way)
        for (int i = 0; i < n; i++) {
            float ax = 0.f, ay = 0.f, az = 0.f;
            for (int j = 0; j < n; j++) {
                float dx = b[j].x - b[i].x;
                float dy = b[j].y - b[i].y;
                float dz = b[j].z - b[i].z;
                float d2 = dx*dx + dy*dy + dz*dz + soft*soft;
                float inv = 1.0f / sqrtf(d2);
                float sc = b[j].mass * inv * inv * inv;
                ax += dx*sc; ay += dy*sc; az += dz*sc;
            }
            b[i].vx += dt*ax; b[i].vy += dt*ay; b[i].vz += dt*az;
        }
        // move particles
        for (int i = 0; i < n; i++) {
            b[i].x += dt*b[i].vx; b[i].y += dt*b[i].vy; b[i].z += dt*b[i].vz;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("CPU: %d bodies, %d steps -> %.2f ms\n", n, steps, ms);

    free(b);
    return 0;
}
