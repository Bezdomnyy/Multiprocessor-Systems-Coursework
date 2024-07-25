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
#define DBG 1
#define TAG 01134

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

  int chunk, start, end, nWorkers;
  int *recvCnt;

  // MASTER: calculate the amount of work for each thread
  if (rank == MASTER) {
    // make the output folder
    mkdir(folder, 0700);
    
    nWorkers = size - 1;
    chunk = (nBodies + nWorkers - 1) / nWorkers;
    recvCnt = (int*)malloc(nWorkers*sizeof(int));

    for (int worker = 1; worker < size; worker++) {
      // calculate the worker's chunk of work
      start = (worker-1) * chunk;
      end = start + chunk < nBodies ? start + chunk : nBodies;

      // keep track of how many elements we need to receive from the current worker
      recvCnt[worker] = (end - start) * 6;

      // send params to worker
      MPI_Send(&chunk, 1, MPI_INT, worker, TAG, MPI_COMM_WORLD); // send chunk
      MPI_Send(&start, 1, MPI_INT, worker, TAG, MPI_COMM_WORLD); // send start
      MPI_Send(&end, 1, MPI_INT, worker, TAG, MPI_COMM_WORLD); // send end
    }
  }
  // WORKER: receive the calculated work parameters
  else {
    // receive work params
    MPI_Status status;
    MPI_Recv(&chunk, 1, MPI_INT, MASTER, TAG, MPI_COMM_WORLD, &status); // recv chunk
    MPI_Recv(&start, 1, MPI_INT, MASTER, TAG, MPI_COMM_WORLD, &status); // recv start
    MPI_Recv(&end, 1, MPI_INT, MASTER, TAG, MPI_COMM_WORLD, &status); // recv end
  }

  // Allocate buffers based on size of work
  int buff_size;
  int bytes;
  float *buf;
  Body *p, *p_master;

  // alocate the buffer where the simulation results will be stored
  bytes = nBodies*sizeof(Body);
  buf = (float*)malloc(bytes);
  p = (Body*) buf;

  // set the initial body coordinates
  // the rng seed should be the same for all workers by default
  // only master does this to save some time
  if (rank == MASTER) randomizeBodies(buf, 6*nBodies);

  // needed for MPI_Recv
  MPI_Status status;

  /* SIMULATE */
  for (int iter = 0; iter < nIters; iter++) {

    /* MASTER */

    if (rank == MASTER) {

      // send simulation data to processes [1]
      for (int worker = 1; worker < size; worker++) {
        MPI_Send(p,nBodies*6,MPI_FLOAT,worker,TAG,MPI_COMM_WORLD);
      }

      // receive simulation data with updated VELOCITIES
      for (int worker = 1; worker < size; worker++) {
        MPI_Recv(&p[(worker-1)*chunk],recvCnt[worker],MPI_FLOAT,worker,TAG,MPI_COMM_WORLD, &status);
      }

      // save the iteration results
      saveToCSV(p, nBodies, iter, folder);

      // send simulation data to processes [2]
      for (int worker = 1; worker < size; worker++) {
        MPI_Send(p,nBodies*6,MPI_FLOAT,worker,TAG,MPI_COMM_WORLD);
      }

      // receive simulation data with updated POSITIONS
      for (int worker = 1; worker < size; worker++) {
        MPI_Recv(&p[(worker-1)*chunk],recvCnt[worker],MPI_FLOAT,worker,TAG,MPI_COMM_WORLD, &status);
      }

      // MASTER does not need to do anything else in this iteration
      continue;

    }

    /* WORKER */

    // receive simulation data [1]
    MPI_Recv(p,nBodies*6,MPI_FLOAT,MASTER,TAG,MPI_COMM_WORLD,&status);

    // the workers update the velocities
    bodyForce(p, dt, nBodies, start, end);

    // return simulation data with updated VELOCITIES
    MPI_Send(&p[start],(end-start)*6,MPI_FLOAT,MASTER,TAG,MPI_COMM_WORLD);

    // receive simulation data [1]
    MPI_Recv(p,nBodies*6,MPI_FLOAT,MASTER,TAG,MPI_COMM_WORLD,&status);

    // the workers update the positions
    for (int i = start ; i < end; i++) { 
      p[i].x += p[i].vx*dt;
      p[i].y += p[i].vy*dt;
      p[i].z += p[i].vz*dt;
    }

    // return simulation data with updated POSITIONS
    MPI_Send(&p[start],(end-start)*6,MPI_FLOAT,MASTER,TAG,MPI_COMM_WORLD);

  }

  // free allocated buffers
  free(buf);
  if (rank == MASTER) free(recvCnt);

  // finalize mpi world
  MPI_Finalize();
}
