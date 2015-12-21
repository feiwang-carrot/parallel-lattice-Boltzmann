#include <stdio.h>
#include <cuda.h>
#include <cublas.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <helper_cuda.h>
#include <helper_timer.h>

const int BLOCK_SIZE_X = 26;  
const int BLOCK_SIZE_Y = 26;
const float w1 = 4.0/9.0, w2 = 1.0/9.0, w3 = 1.0/36.0;
const float Amp2 = 0.1, Width = 10, omega = 1;


 __global__ void Denrho(float* u_d, float* f_d, int ArraySizeX, int ArraySizeY)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x*(BLOCK_SIZE_X-2);
  int by = blockIdx.y*(BLOCK_SIZE_Y-2);
  int x = tx + bx;
  int y = ty + by;
  u_d[x*ArraySizeY+y] = 0;
  for (int i=0;i<9;i++)
  u_d[x*ArraySizeY+y] += (float)f_d[x*ArraySizeY*9+y*9+i];
  
  __syncthreads();
} 

__global__ void iteration(float* f_d, int ArraySizeX, int ArraySizeY)
{ 
  int i;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x*(BLOCK_SIZE_X-2);
  int by = blockIdx.y*(BLOCK_SIZE_Y-2);
  int x = tx + bx;
  int y = ty + by;
  register float n,ux,uy,uxx,uyy,uxy,usq,Fx,Fy,Fxx,Fyy,Fxy,Fsq;
  __shared__ float f_sh[BLOCK_SIZE_X][BLOCK_SIZE_Y][9];
//  __shared__ float f_p[BLOCK_SIZE_X][BLOCK_SIZE_Y][9];

  for(i=0;i<9;i++)
  f_sh[tx][ty][i]=f_d[x*ArraySizeY*9+y*9+ i];

  __syncthreads();

  n=f_sh[tx][ty][0]+f_sh[tx][ty][1]+f_sh[tx][ty][2]+f_sh[tx][ty][3]+f_sh[tx][ty][4]+f_sh[tx][ty][5]+f_sh[tx][ty][6]+f_sh[tx][ty][7]+f_sh[tx][ty][8];
  ux=f_sh[tx][ty][1]-f_sh[tx][ty][2]+f_sh[tx][ty][5]-f_sh[tx][ty][6]-f_sh[tx][ty][7]+f_sh[tx][ty][8];
  uy=f_sh[tx][ty][3]-f_sh[tx][ty][4]+f_sh[tx][ty][5]+f_sh[tx][ty][6]-f_sh[tx][ty][7]-f_sh[tx][ty][8];
  ux/=n;
  uy/=n;
  uxx=ux*ux;
  uyy=uy*uy;
  uxy=2*ux*uy;
  usq=uxx+uyy;
  // implement the forcing terms and perform collision step
  Fx=0;//Amp*sin(y*2*M_PI/cols);
  Fy=0;
  Fxx=2*n*Fx*ux;
  Fyy=2*n*Fy*uy;
  Fxy=2*n*(Fx*uy+Fy*ux);
  Fsq=Fxx+Fyy;
  Fx*=n;
  Fy*=n;
   
  f_sh[tx][ty][0]+=omega*(w1*n*(1-1.5*usq)-f_sh[tx][ty][0])-w1*1.5*Fsq;
  f_sh[tx][ty][1]+=omega*(w2*n*(1+3*ux+4.5*uxx -1.5*usq)-f_sh[tx][ty][1])+w2*(3*Fx+4.5*Fxx-1.5*Fsq);
  f_sh[tx][ty][2]+=omega*(w2*n*(1-3*ux+4.5*uxx -1.5*usq)-f_sh[tx][ty][2])+w2*(-3*Fx+4.5*Fxx-1.5*Fsq);
  f_sh[tx][ty][3]+=omega*(w2*n*(1+3*uy+4.5*uyy -1.5*usq)-f_sh[tx][ty][3])+w2*(3*Fy+4.5*Fyy-1.5*Fsq);
  f_sh[tx][ty][4]+=omega*(w2*n*(1-3*uy+4.5*uyy -1.5*usq)-f_sh[tx][ty][4])+w2*(-3*Fy+4.5*Fyy-1.5*Fsq);
  f_sh[tx][ty][5]+=omega*(w3*n*(1+3*(ux+uy)+4.5*(uxx+uxy+uyy)-1.5*usq)-f_sh[tx][ty][5])+w3*(3*(Fx+Fy)+4.5*(Fxx+Fxy+Fyy)-1.5*Fsq);
  f_sh[tx][ty][6]+=omega*(w3*n*(1+3*(-ux+uy)+4.5*(uxx-uxy+uyy)-1.5*usq)-f_sh[tx][ty][6])+w3*(3*(-Fx+Fy)+4.5*(Fxx-Fxy+Fyy)-1.5*Fsq);
  f_sh[tx][ty][7]+=omega*(w3*n*(1+3*(-ux-uy)+4.5*(uxx+uxy+uyy)-1.5*usq)-f_sh[tx][ty][7])+w3*(3*(-Fx-Fy)+4.5*(Fxx+Fxy+Fyy)-1.5*Fsq);
  f_sh[tx][ty][8]+=omega*(w3*n*(1+3*(ux-uy)+4.5*(uxx-uxy+uyy)-1.5*usq)-f_sh[tx][ty][8])+w3*(3*(Fx-Fy)+4.5*(Fxx-Fxy+Fyy)-1.5*Fsq);
  __syncthreads();
  
  //perfom stream step
  if(tx>0 && tx<BLOCK_SIZE_X-1 && ty>0 && ty<BLOCK_SIZE_Y-1) {
  f_d[x*ArraySizeY*9+y*9] = f_sh[tx][ty][0];//+omega*(w1*n*(1-1.5*usq)-f_sh[tx][ty][0])-w1*1.5*Fsq;
  f_d[x*ArraySizeY*9+y*9+2] = f_sh[tx+1][ty][2];//+omega*(w2*n*(1-3*ux+4.5*uxx -1.5*usq)-f_sh[tx+1][ty][2])+w2*(-3*Fx+4.5*Fxx-1.5*Fsq);
  f_d[x*ArraySizeY*9+y*9+1] = f_sh[tx-1][ty][1];//+omega*(w2*n*(1+3*ux+4.5*uxx -1.5*usq)-f_sh[tx-1][ty][1])+w2*(3*Fx+4.5*Fxx-1.5*Fsq);
  f_d[x*ArraySizeY*9+y*9+4] = f_sh[tx][ty+1][4];//+omega*(w2*n*(1-3*uy+4.5*uyy -1.5*usq)-f_sh[tx][ty+1][4])+w2*(-3*Fy+4.5*Fyy-1.5*Fsq);
  f_d[x*ArraySizeY*9+y*9+3] = f_sh[tx][ty-1][3];//+omega*(w2*n*(1+3*uy+4.5*uyy -1.5*usq)-f_sh[tx][ty-1][3])+w2*(3*Fy+4.5*Fyy-1.5*Fsq);
  f_d[x*ArraySizeY*9+y*9+7] = f_sh[tx+1][ty+1][7];//+omega*(w3*n*(1+3*(-ux-uy)+4.5*(uxx+uxy+uyy)-1.5*usq)-f_sh[tx+1][ty+1][7])+w3*(3*(-Fx-Fy)+4.5*(Fxx+Fxy+Fyy)-1.5*Fsq);
  f_d[x*ArraySizeY*9+y*9+5] = f_sh[tx-1][ty-1][5];//+omega*(w3*n*(1+3*(ux+uy)+4.5*(uxx+uxy+uyy)-1.5*usq)-f_sh[tx-1][ty-1][5])+w3*(3*(Fx+Fy)+4.5*(Fxx+Fxy+Fyy)-1.5*Fsq);
  f_d[x*ArraySizeY*9+y*9+6] = f_sh[tx+1][ty-1][6];//+omega*(w3*n*(1+3*(-ux+uy)+4.5*(uxx-uxy+uyy)-1.5*usq)-f_sh[tx+1][ty-1][6])+w3*(3*(-Fx+Fy)+4.5*(Fxx-Fxy+Fyy)-1.5*Fsq);
  f_d[x*ArraySizeY*9+y*9+8] = f_sh[tx-1][ty+1][8];//+omega*(w3*n*(1+3*(ux-uy)+4.5*(uxx-uxy+uyy)-1.5*usq)-f_sh[tx-1][ty+1][8])+w3*(3*(Fx-Fy)+4.5*(Fxx-Fxy+Fyy)-1.5*Fsq);
  
  } 
 
   __syncthreads();
   // apply periodi boundary conditions;
  if(x == 0)
   for(i = 0;i<9;i++)
    f_d[x*ArraySizeY*9+y*9+i] = f_d[(ArraySizeX-2)*ArraySizeY*9+y*9+i];
  if(x==ArraySizeX-1)
   for(i=0;i<9;i++)
    f_d[x*ArraySizeY*9+y*9+i] = f_d[ArraySizeY*9+y*9+i];
  if(y == 0)
   for(i = 0;i<9;i++)
    f_d[x*ArraySizeY*9+y*9+i] = f_d[x*ArraySizeY*9+(ArraySizeY-2)*9+i];
  if(y == ArraySizeY-1)
   for(i =0;i<9;i++)
    f_d[x*ArraySizeY*9 +y*9 +i] = f_d[x*ArraySizeY*9+9+i];
    
  //   __syncthreads(); 
}


int main(int argc, char **argv)
{ float *f_d, *f_h;    // pointers to host memory
  float *u_d, *u_h;    // pointers to device memory
  int ArraySizeX=5162;  // Note these, minus boundary layer, have to be exactly divisible by the (BLOCK_SIZE-2) here
  int ArraySizeY=5162;
  size_t size=ArraySizeX*ArraySizeY*sizeof(float);
  size_t size1 = ArraySizeX*ArraySizeY*sizeof(float);
  FILE *fp;
  float n,ux,uy,uxx,uyy,uxy,usq;
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  // CUT_DEVICE_INIT(argc, argv);
  // CUT_SAFE_CALL(cutCreateTimer(&hTimer));
  if(error_id != cudaSuccess)
  { printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
    exit(EXIT_FAILURE);
  } 
  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0)
   {
   printf("There are no available device(s) that support CUDA\n");
   exit(EXIT_FAILURE);
   }
   else
   {
    printf("Detected %d CUDA Capable device(s)\n", deviceCount);
   }
  for (int dev = 0; dev < deviceCount; ++dev)
   {
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf(" Total amount of shared memory per block: %lu bytes\n", deviceProp.sharedMemPerBlock);
    printf(" Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    if(BLOCK_SIZE_X*BLOCK_SIZE_Y > deviceProp.maxThreadsPerBlock)
       exit(EXIT_FAILURE);
   }
  //Allocate arrays on host and initialize to zero
  sdkStartTimer(&timer);
  //CUT_SAFE_CALL(cutResetTimer(hTimer));
  //CUT_SAFE_CALL(cutStartTimer(hTimer));
  u_h=(float *)calloc(ArraySizeX*ArraySizeY,sizeof(float));
  f_h=(float *)calloc(ArraySizeX*ArraySizeY*9,sizeof(float));
  //Allocate arrays on device
  cudaMalloc((void **) &u_d,size1);
  cudaMalloc((void **) &f_d,9*size);
  // initialization 
  for(int x = 0;x<ArraySizeX;x++){
     for(int y =0;y<ArraySizeY;y++){
	// define the macroscopic properties of the initial condition.
     n = 1 + Amp2*exp(-(pow(x-ArraySizeX/2,2)+pow(y-ArraySizeY/2,2))/Width);
     ux = 0;
     uy = 0;		
      // intialize f to be the local equilibrium values	
     uxx = ux*ux;
     uyy = uy*uy;
     uxy = 2*ux*uy;
     usq = uxx+ uyy;
	  
     f_h[x*ArraySizeY*9+y*9] = w1*n*(1-1.5*usq);
     f_h[x*ArraySizeY*9+y*9+1] = w2*n*(1+3*ux+4.5*uxx-1.5*usq);
     f_h[x*ArraySizeY*9+y*9+2] = w2*n*(1-3*ux+4.5*uxx-1.5*usq);
     f_h[x*ArraySizeY*9+y*9+3] = w2*n*(1+3*uy+4.5*uyy-1.5*usq);
     f_h[x*ArraySizeY*9+y*9+4]= w2*n*(1-3*uy+4.5*uyy-1.5*usq); 
     f_h[x*ArraySizeY*9+y*9+5] = w3*n*(1+3*(ux+uy)+4.5*(uxx+uxy+uyy)-1.5*usq);
     f_h[x*ArraySizeY*9+y*9+6] = w3*n*(1+3*(-ux+uy)+4.5*(uxx-uxy+uyy)-1.5*usq);
     f_h[x*ArraySizeY*9+y*9+7] = w3*n*(1+3*(-ux-uy)+4.5*(uxx+uxy+uyy)-1.5*usq);
     f_h[x*ArraySizeY*9+y*9+8] = w3*n*(1+3*(ux-uy)+4.5*(uxx-uxy+uyy)-1.5*usq);
	}
    }

 //  cudaMemcpy(u_d,u_h,size, cudaMemcpyHostToDevice);
  cudaMemcpy(f_d,f_h,9*size, cudaMemcpyHostToDevice);

  // Part 2 of 4: Set up execution configuration
  int nBlocksX=(ArraySizeX-2)/(BLOCK_SIZE_X-2);
  int nBlocksY=(ArraySizeY-2)/(BLOCK_SIZE_Y-2);

  dim3 dimBlock(BLOCK_SIZE_X,BLOCK_SIZE_Y);
  dim3 dimGrid(nBlocksX,nBlocksY);

  for (int nsteps=0; nsteps <100; nsteps++) {
    // Part 3 of 4: Call kernel with execution configuration
    iteration<<<dimGrid, dimBlock>>>(f_d,ArraySizeX,ArraySizeY);
    
   // cudaMemcpy(&fd[0],&fd[ArraySize-2],) 
  }   
  
  Denrho<<<dimGrid, dimBlock>>>(u_d,f_d,ArraySizeX,ArraySizeY);
 // Part 4 of 4: Retrieve result from device and store in u_h
  cudaMemcpy(u_h, u_d, size1, cudaMemcpyDeviceToHost);
  sdkStopTimer(&timer);
  float retval = sdkGetTimerValue(&timer);
  // CUT_SAFE_CALL(cutStopTimer(hTimer));
   // gputime = cutGetTimerValue(hTimer);
  int ind=cublasIsamax(ArraySizeX*ArraySizeY,u_d,1);
  float avg= cublasSasum(ArraySizeX*ArraySizeY,u_d,1)/(ArraySizeX*ArraySizeY);
  float value = u_h[ind-1];
  printf("max density is %f, average density is %f, gputime is %f", value, avg,0.001*retval);
  // Output results
  fp = fopen("SolutionGPU50.txt","wt");

  for (int i=0; i<ArraySizeX; i++) {
    for (int j=0; j<ArraySizeY; j++)
      fprintf(fp," %6.4f",u_h[i*ArraySizeY+j]);
    fprintf(fp,"\n");
  }

  fclose(fp);
  //Cleanup
  free(u_h);
  free(f_h);
  cudaFree(u_d);
  cudaFree(f_d);
}
