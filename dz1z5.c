#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <omp.h>

#define SOFTENING 1e-9f

typedef struct { float x, y, z, vx, vy, vz; } Body;

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

void bodyForce(Body *p, float dt, int n) {
#pragma omp for
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

int main(const int argc, const char** argv) {
  
  int nBodies = atoi(argv[1]);
  int nIters = atoi(argv[2]);
  const char *folder = argv[3];

  const float dt = 0.01f; 

  mkdir(folder, 0700);

  int bytes = nBodies*sizeof(Body);
  float *buf = (float*)malloc(bytes);
  Body *p = (Body*)buf;

  //srand(1);

  randomizeBodies(buf, 6*nBodies);

  // double start_time, end_time, elapsed_time;
  // start_time = omp_get_wtime();

#pragma omp parallel
  bodyForce(p, dt, nBodies);

  saveToCSV(p, nBodies, 0, folder); 

  for (int iter = 1; iter < nIters; iter++) {

#pragma omp parallel
{
#pragma omp for
    for (int i = 0 ; i < nBodies; i++) { 
      p[i].x += p[i].vx*dt;
      p[i].y += p[i].vy*dt;
      p[i].z += p[i].vz*dt;
    }

    bodyForce(p, dt, nBodies);
}

    saveToCSV(p, nBodies, iter, folder); 

  }

  // end_time = omp_get_wtime();
  // elapsed_time = end_time - start_time;
  // printf ("\nParallelised section execution time: %f\n\n", elapsed_time);

  free(buf);
}
