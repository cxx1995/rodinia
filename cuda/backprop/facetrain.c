

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "backprop.h"
#include "omp.h"

extern char *strcpy();
extern void exit();

int layer_size = 0;

backprop_face()
{
  BPNN *net;
  int i;
  float out_err, hid_err;
  net = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)
  
  printf("Input layer size : %d\n", layer_size);
  load(net);
  //entering the training kernel, only one iteration
  printf("Starting training kernel\n");
  struct timeval start;
  struct timeval end;
  double diff;
  gettimeofday(&start,NULL);
  bpnn_train_cuda(net, &out_err, &hid_err);
  gettimeofday(&end,NULL);
  diff = (1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec)/(double)1000000;
  printf("The Duration is: %f\n",diff);

  bpnn_free(net);
  printf("Training done\n");
}

int setup(argc, argv)
int argc;
char *argv[];
{
	
  int seed;

  if (argc!=2){
  fprintf(stderr, "usage: backprop <num of input elements>\n");
  exit(0);
  }
  layer_size = atoi(argv[1]);
  if (layer_size%16!=0){
  fprintf(stderr, "The number of input points must be divided by 16\n");
  exit(0);
  }
  
  printf("Start\n");
  seed = 7;   
  bpnn_initialize(seed);
  backprop_face();

  exit(0);
}
