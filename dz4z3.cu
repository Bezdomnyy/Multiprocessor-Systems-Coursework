#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#define SOFTENING 1e-9f

#define BLOCK_SIZE 1024

typedef struct { float x, y, z, vx, vy, vz; } Body;

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

void bodyForce(Body *p, float dt, int n) {
  for (int i = 0; i < n; i++) { 
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = 1.0f / sqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }

    p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
  }
}

void saveToCSV(Body *p, int n, int iter, const char *folder) {
  char filename[50];
  sprintf(filename, "%s/iteration_%d.csv", folder, iter);
  FILE *file = fopen(filename, "w");

  fprintf(file, "x,y,z,vx,vy,vz\n");
  for (int i = 0; i < n; i++) {
    fprintf(file, "%f,%f,%f,%f,%f,%f\n", p[i].x, p[i].y, p[i].z, p[i].vx, p[i].vy, p[i].vz);
  }

  fclose(file);
}

__global__ void bodyForceKernel(Body *p, float dt, int n) {

  int myIdx = blockIdx.x * blockDim.x + threadIdx.x;

  float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

  if (myIdx < n) {
    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[myIdx].x;
      float dy = p[j].y - p[myIdx].y;
      float dz = p[j].z - p[myIdx].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = 1.0f / sqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }

    p[myIdx].vx += dt*Fx; p[myIdx].vy += dt*Fy; p[myIdx].vz += dt*Fz;
  }

}

__global__ void bodyPositionKernel(Body *p, float dt, int n) {
  int myIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (myIdx < n) {
    p[myIdx].x += p[myIdx].vx*dt;
    p[myIdx].y += p[myIdx].vy*dt;
    p[myIdx].z += p[myIdx].vz*dt;
  }
}



int main(const int argc, const char** argv) {
  
  int nBodies = atoi(argv[1]);
  int nIters = atoi(argv[2]);
  const char *folder = argv[3];

  const float dt = 0.01f; 

  mkdir(folder, 0700);

  dim3 dimGrid((nBodies + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 dimBlock(BLOCK_SIZE);

  int bytes = nBodies*sizeof(Body);
  float *buf = (float*)malloc(bytes);
  float *gpu_buf;
  cudaMalloc(&gpu_buf, bytes);
  Body *p = (Body*)buf;
  Body *gpu_p = (Body*)gpu_buf;

  randomizeBodies(buf, 6*nBodies);

  //cudaMemcpy(gpu_buf, buf, bytes, cudaMemcpyHostToDevice);

  for (int iter = 0; iter < nIters; iter++) {

    cudaMemcpy(gpu_buf, buf, bytes, cudaMemcpyHostToDevice);

    bodyForceKernel<<<dimGrid,dimBlock>>>(gpu_p, dt, nBodies);

    cudaMemcpy(buf, gpu_buf, bytes, cudaMemcpyDeviceToHost);

    saveToCSV(p, nBodies, iter, folder);

    // bodyPositionKernel<<<dimGrid, dimBlock>>>(gpu_p, dt, nBodies);

    for (int i = 0 ; i < nBodies; i++) { 
      p[i].x += p[i].vx*dt;
      p[i].y += p[i].vy*dt;
      p[i].z += p[i].vz*dt;
    }

  }

  free(buf);
}
