#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#define SOFTENING 1e-9f

typedef struct { float x, y, z, vx, vy, vz; } Body;

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

void bodyForce(Body *p, float dt, int n, int start, int end) {
  // printf("start:%d; end:%d nBodies:%d\n",start, end, n);
  for (int i = start; i < end; i++) { 
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


#define MASTER 0

int main(int argc, char** argv) {
  int dbg_i = 0;

  // init mpi world
  MPI_Init(&argc, &argv);

  // init mpi comm vars
  int size, rank=0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  // init the simulation params
  int nBodies = atoi(argv[1]);
  int nIters = atoi(argv[2]);
  const char *folder = argv[3];

  const float dt = 0.01f;

  // make the output folder
  mkdir(folder, 0700);

  // calculate the amount of work for each thread
  int chunk = (nBodies + size - 1) / size;
  int start = rank * chunk;
  int end = start + chunk < nBodies ? start + chunk : nBodies;

  // DOUBLE BUFFERING:
  //
  // - p first points to buf0
  // - when MPI_Allgather is executed, data is sent from buf0 and gathered in buf1
  // - p is then changed to point to buf1 for the next iteration and the process is repeated
  //
  // - ordering of operations is changed so the calculations can be done with less communicational overhead
  //
  int bytes = chunk*size*sizeof(Body);
  float *buf0 = (float*)malloc(bytes);
  float *buf1 = (float*)malloc(bytes);
  Body *p[2];
  p[0] = (Body*) buf0;
  p[1] = (Body*) buf1;

  // a variable to select the correct buffer for the iteration
  int selected = 0;

  // set the initial body coordinates
  // the rng seed should be the same for all workers by default
  randomizeBodies(buf0, 6*nBodies);

  for (int iter = 0; iter < nIters; iter++) {
    // update forces
    bodyForce(p[selected], dt, nBodies, start, end);

    // receive the chunks calculated by other workers
    MPI_Allgather(&p[selected][start],6*chunk,MPI_FLOAT,p[1-selected],6*chunk,MPI_FLOAT,MPI_COMM_WORLD);

    // switch the send and receive buffers
    selected = 1 - selected;

    // save the iteration results
    if (rank == MASTER)
      saveToCSV(p[selected], nBodies, iter, folder);

    // update positions
    for (int i = start ; i < end; i++) { 
      p[selected][i].x += p[selected][i].vx*dt;
      p[selected][i].y += p[selected][i].vy*dt;
      p[selected][i].z += p[selected][i].vz*dt;
    }

    // receive the chunks calculated by other workers
    MPI_Allgather(&p[selected][start],6*chunk,MPI_FLOAT,p[1-selected],6*chunk,MPI_FLOAT,MPI_COMM_WORLD);

    // switch the send and receive buffers
    selected = 1 - selected;

  }

  free(buf0);
  free(buf1);

  // finalize mpi world
  MPI_Finalize();
}
