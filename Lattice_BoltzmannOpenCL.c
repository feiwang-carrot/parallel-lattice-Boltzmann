#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

const int BLOCK_SIZE_X = 18; //interior +2 bounday 
const int BLOCK_SIZE_Y = 18;
const double w1 = 4.0/9.0, w2 = 1.0/9.0, w3 = 1.0/36.0;
const double Amp2 = 0.1, Width = 10, omega = 1;

const char* programSource =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable \n"
"__kernel void Denrho(__global float* u_d, __global double* f_d, int ArraySizeX, int ArraySizeY)  \n"
"{                                                                                                \n"
"  int i;                                                                                      \n"
"  int blk_size_x = get_local_size(0);                                                         \n"
"  int blk_size_y = get_local_size(1);                                                         \n"
"  int tx = get_local_id(0);                                                                   \n"
"  int ty = get_local_id(1);                                                                   \n"
"  int bx = get_group_id(0)*(blk_size_x-2);                                                    \n"
"  int by = get_group_id(1)*(blk_size_y-2);                                                    \n"
"  int x = tx + bx;                                                                            \n"
"  int y = ty + by;                                                                            \n"
"  u_d[x*ArraySizeY+y] = 0;                                                                    \n"
"  for(i = 0;i<9;i++)                                                                          \n"
"  u_d[x*ArraySizeY+y] += (float)f_d[x*ArraySizeY*9+y*9+i];                                    \n"
"  barrier(CLK_LOCAL_MEM_FENCE);                                                               \n"
"}                                                                                             \n"
// const char* programSource1=
// "#pragma OPENCL EXTENSION cl_khr_fp64 : enable                                                 \n"
"__kernel void lbiteration(__global double* f_d, int ArraySizeX, int ArraySizeY)               \n"
"{                                                                                             \n"
"  int i;                                                                                      \n"
"  double omega = 1,w1 = 4.0/9.0, w2 = 1.0/9.0, w3 = 1.0/36.0;                                 \n"
"  int blk_size_x = get_local_size(0);                                                         \n"
"  int blk_size_y = get_local_size(1);                                                         \n"
"  int tx = get_local_id(0);                                                              \n"
"  int ty = get_local_id(1);                                                              \n"
"  int bx = get_group_id(0)*(blk_size_x -2);                                                   \n"
"  int by = get_group_id(1)*(blk_size_y -2);                                                   \n"
"  int x = tx + bx;                                                                            \n"
"  int y = ty + by;                                                                            \n"
"  register double n, ux, uy, uxx, uyy, uxy, usq,Fx,Fy,Fxx,Fyy,Fxy,Fsq;                        \n"
"  __local double f_sh[18][18][9];                                                             \n"
"  for(i = 0;i<9;i++)                                                                          \n"
"  f_sh[tx][ty][i] = f_d[x*ArraySizeY*9 + y*9 +i];                                             \n"
"  barrier(CLK_LOCAL_MEM_FENCE);                                                               \n"
"  n = f_sh[tx][ty][0] + f_sh[tx][ty][1] + f_sh[tx][ty][2] + f_sh[tx][ty][3] + f_sh[tx][ty][4] + f_sh[tx][ty][5] + f_sh[tx][ty][6]+f_sh[tx][ty][7] +f_sh[tx][ty][8]; \n"
"  ux=f_sh[tx][ty][1]-f_sh[tx][ty][2]+f_sh[tx][ty][5]-f_sh[tx][ty][6]-f_sh[tx][ty][7]+f_sh[tx][ty][8]; \n"
"  uy=f_sh[tx][ty][3]-f_sh[tx][ty][4]+f_sh[tx][ty][5]+f_sh[tx][ty][6]-f_sh[tx][ty][7]-f_sh[tx][ty][8]; \n"
"  ux /= n;   \n"
"  uy /= n;   \n"
"  uxx = ux*ux; \n"
"  uyy = uy*uy; \n"
"  uxy = 2*ux*uy; \n"
"  usq = uxx + uyy; \n"
"  Fx=0;  \n"
"  Fy=0;   \n"
"  Fxx=2*n*Fx*ux; \n"
"  Fyy=2*n*Fy*uy; \n"
"  Fxy=2*n*(Fx*uy+Fy*ux); \n"
"  Fsq=Fxx+Fyy;  \n"
"  Fx*=n;   \n"
"  Fy*=n;   \n"   
"  f_sh[tx][ty][0]+=omega*(w1*n*(1-1.5*usq)-f_sh[tx][ty][0])-w1*1.5*Fsq; \n"
"  f_sh[tx][ty][1]+=omega*(w2*n*(1+3*ux+4.5*uxx -1.5*usq)-f_sh[tx][ty][1])+w2*(3*Fx+4.5*Fxx-1.5*Fsq); \n"
"  f_sh[tx][ty][2]+=omega*(w2*n*(1-3*ux+4.5*uxx -1.5*usq)-f_sh[tx][ty][2])+w2*(-3*Fx+4.5*Fxx-1.5*Fsq); \n"
"  f_sh[tx][ty][3]+=omega*(w2*n*(1+3*uy+4.5*uyy -1.5*usq)-f_sh[tx][ty][3])+w2*(3*Fy+4.5*Fyy-1.5*Fsq);  \n"
"  f_sh[tx][ty][4]+=omega*(w2*n*(1-3*uy+4.5*uyy -1.5*usq)-f_sh[tx][ty][4])+w2*(-3*Fy+4.5*Fyy-1.5*Fsq); \n"
"  f_sh[tx][ty][5]+=omega*(w3*n*(1+3*(ux+uy)+4.5*(uxx+uxy+uyy)-1.5*usq)-f_sh[tx][ty][5])+w3*(3*(Fx+Fy)+4.5*(Fxx+Fxy+Fyy)-1.5*Fsq); \n"
"  f_sh[tx][ty][6]+=omega*(w3*n*(1+3*(-ux+uy)+4.5*(uxx-uxy+uyy)-1.5*usq)-f_sh[tx][ty][6])+w3*(3*(-Fx+Fy)+4.5*(Fxx-Fxy+Fyy)-1.5*Fsq); \n"
"  f_sh[tx][ty][7]+=omega*(w3*n*(1+3*(-ux-uy)+4.5*(uxx+uxy+uyy)-1.5*usq)-f_sh[tx][ty][7])+w3*(3*(-Fx-Fy)+4.5*(Fxx+Fxy+Fyy)-1.5*Fsq); \n"
"  f_sh[tx][ty][8]+=omega*(w3*n*(1+3*(ux-uy)+4.5*(uxx-uxy+uyy)-1.5*usq)-f_sh[tx][ty][8])+w3*(3*(Fx-Fy)+4.5*(Fxx-Fxy+Fyy)-1.5*Fsq);  \n"
"   barrier(CLK_LOCAL_MEM_FENCE);   \n"
"  //perfom stream step \n"
"  if(tx>0 && tx<17 && ty>0 && ty<17) {   \n"
"  f_d[x*ArraySizeY*9+y*9] = f_sh[tx][ty][0];   \n"
"  f_d[x*ArraySizeY*9+y*9+2] = f_sh[tx+1][ty][2];  \n"
"  f_d[x*ArraySizeY*9+y*9+1] = f_sh[tx-1][ty][1];  \n"
"  f_d[x*ArraySizeY*9+y*9+4] = f_sh[tx][ty+1][4];  \n"
"  f_d[x*ArraySizeY*9+y*9+3] = f_sh[tx][ty-1][3];  \n"
"  f_d[x*ArraySizeY*9+y*9+7] = f_sh[tx+1][ty+1][7]; \n"
"  f_d[x*ArraySizeY*9+y*9+5] = f_sh[tx-1][ty-1][5]; \n"
"  f_d[x*ArraySizeY*9+y*9+6] = f_sh[tx+1][ty-1][6]; \n"
"  f_d[x*ArraySizeY*9+y*9+8] = f_sh[tx-1][ty+1][8]; \n"
"  } \n" 
"  barrier(CLK_LOCAL_MEM_FENCE);  \n"
"   // apply periodi boundary conditions;  \n"
"  if(x == 0)  \n"
"   for(i = 0;i<9;i++)   \n"
"    f_d[x*ArraySizeY*9+y*9+i] = f_d[(ArraySizeX-2)*ArraySizeY*9+y*9+i];  \n"
"  if(x==ArraySizeX-1)  \n"
"   for(i=0;i<9;i++)   \n"
"    f_d[x*ArraySizeY*9+y*9+i] = f_d[ArraySizeY*9+y*9+i];   \n"
"  if(y == 0)  \n"
"   for(i = 0;i<9;i++)   \n"
"    f_d[x*ArraySizeY*9+y*9+i] = f_d[x*ArraySizeY*9+(ArraySizeY-2)*9+i];   \n"
"  if(y == ArraySizeY-1)   \n"
"   for(i =0;i<9;i++)   \n"
"    f_d[x*ArraySizeY*9 +y*9 +i] = f_d[x*ArraySizeY*9+9+i];   \n"
"  barrier(CLK_LOCAL_MEM_FENCE);   \n"    
"}    \n" 
;
void chk(cl_int status, const char* cmd){
  if(status != CL_SUCCESS){
    printf("%s failed (%d) \n", cmd, status);
     exit(-1);
   }
}

int main(int argc, char **argv)
{  
   printf("start \n");
   int x, y, nsteps, i, j;
   float *u_h;
   double *f_h;  //pointers to host memory	
   int ArraySizeX = 5122;
   int ArraySizeY = 5122;
   double n, ux, uy, uxx, uxy, uyy, usq;
   FILE *fp;	
   size_t size = ArraySizeX*ArraySizeY*sizeof(float);
   size_t size1 = ArraySizeX*ArraySizeY*9*sizeof(double);
   u_h = (float *)calloc(ArraySizeX*ArraySizeY,sizeof(float));
   f_h = (double *)calloc(ArraySizeX*ArraySizeY*9,sizeof(double));
   printf("initialization \n");
    // initialization 
   for( x = 0;x<ArraySizeX;x++){
     for( y =0;y<ArraySizeY;y++){
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
    
     cl_event event;
     cl_ulong time_start, time_end, total_time; 
     // use this to check the output of each API call
     cl_int status;
     // retrieve the number of platforms
     cl_uint numPlatforms = 0;
     status = clGetPlatformIDs(0,NULL,&numPlatforms);
     chk(status, "clGetPlatformIDs0");

     // allocate enough space for each platform
     cl_platform_id *platforms = NULL;
     platforms = (cl_platform_id *) malloc(numPlatforms*sizeof(cl_platform_id));

     // Fill in the platforms
     status = clGetPlatformIDs(numPlatforms, platforms, NULL);    
     chk(status, "clGetPlatformIDs1");

     // Retrieve the number of devices
     cl_uint numDevices = 0;
     status = clGetDeviceIDs(platforms[0],CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
     chk(status, "clGetDeviceIDs0");
  
     // Allocate enough space for each device
     cl_device_id *devices = NULL;
     devices = (cl_device_id *) malloc(numDevices*sizeof(cl_device_id));

     // Fill in the devices
     status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
     chk(status, "clGetDeviceIDs1");

     // Create a context and associate it with devices
     cl_context	 context;
     context = clCreateContext(NULL,numDevices, devices, NULL, NULL, &status);
     chk(status,"clCreateContext");

     // Create  a command queue and associate it with device
     cl_command_queue cmdQueue;
     cmdQueue = clCreateCommandQueue(context, devices[0],CL_QUEUE_PROFILING_ENABLE,&status);
     chk(status,"clCreateCommandQueue");
     
     // Create Buffer objects on devices
     cl_mem u_d, f_d;
     u_d = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &status);
     chk(status,"clCreatebuffer");
     f_d = clCreateBuffer(context, CL_MEM_READ_WRITE, size1, NULL, &status);
     chk(status, "clCreatebuffer");

     // perform computing on GPU
     // copy data from host to device
     status = clEnqueueWriteBuffer(cmdQueue, u_d, CL_FALSE, 0, size, u_h, 0, NULL, NULL);
     chk(status,"ClEnqueueWriteBuffer");
     status = clEnqueueWriteBuffer(cmdQueue, f_d, CL_FALSE, 0, size1, f_h, 0, NULL, NULL);
     chk(status, "clEnqueueWriteBuffer");
     
     // create program with source code
     cl_program program = clCreateProgramWithSource(context,1,(const char**)&programSource, NULL, &status);
     chk(status, "clCreateProgramWithSource");

     // Compile program for the device
     status = clBuildProgram(program, numDevices, devices, NULL, NULL,NULL);
      // chk(status, "ClBuildProgram");
      if(status != CL_SUCCESS){
      printf("clBuildProgram failed (%d) \n", status);
      size_t log_size;
      clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
      
      char *log = (char *) malloc(log_size);
      clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
      printf("%s\n", log);
      exit(-1);
     }
      printf("successfully built program \n");
      
     // Create lattice-boltzman kernel
     cl_kernel kernel, kernel1;
     kernel = clCreateKernel(program, "lbiteration", &status);
     kernel1 = clCreateKernel(program, "Denrho", &status);
     chk(status, "clCreateKernel");
      printf("successfully create kernel \n");
     
     // Associate the input and output buffers with the kernel
     status = clSetKernelArg(kernel,0, sizeof(cl_mem), &f_d);
     status |= clSetKernelArg(kernel1,0, sizeof(cl_mem), &u_d);
     status |= clSetKernelArg(kernel1,1, sizeof(cl_mem), &f_d);
     status |= clSetKernelArg(kernel, 1, sizeof(int), &ArraySizeX);
     status |= clSetKernelArg(kernel1,2, sizeof(int), &ArraySizeX);
     status |= clSetKernelArg(kernel, 2, sizeof(int), &ArraySizeY);
     status |= clSetKernelArg(kernel1,3, sizeof(int),&ArraySizeY);
     chk(status, "clSerKernelArg");
    
     // set the work dimensions
     size_t localworksize[2] = {BLOCK_SIZE_X,BLOCK_SIZE_Y};
     int nBLOCKSX = (ArraySizeX-2)/(BLOCK_SIZE_X -2);
     int nBLOCKSY = (ArraySizeY-2)/(BLOCK_SIZE_Y -2);
     size_t globalworksize[2] = {nBLOCKSX*BLOCK_SIZE_X,nBLOCKSY*BLOCK_SIZE_Y};

     // loop the kernel
     for( nsteps = 0; nsteps < 100; nsteps++){
     status = clEnqueueNDRangeKernel(cmdQueue, kernel, 2, NULL, globalworksize,localworksize,0,NULL,&event);
     clWaitForEvents(1 , &event);
     clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
           sizeof(time_start), &time_start, NULL);
     clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
           sizeof(time_end), &time_end, NULL);
     total_time += time_end - time_start;
     }
     printf("Good so far \n");
     status = clEnqueueNDRangeKernel(cmdQueue, kernel1, 2, NULL, globalworksize,localworksize,0,NULL,&event);
     chk(status, "clEnqueueNDR");
     clWaitForEvents(1 , &event);
     clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
           sizeof(time_start), &time_start, NULL);
     clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
           sizeof(time_end), &time_end, NULL);
     total_time += time_end - time_start;
     printf("running time is %0.3f \n",(total_time/1000000000.0));
     // retrieve data from device
     status = clEnqueueReadBuffer(cmdQueue, u_d, CL_TRUE, 0, size, u_h, 0, NULL, NULL);
     chk(status, "clEnqueueReadBuffer");

     // Output results
     fp = fopen("SolutionCL.txt", "wt");
     for(i= 0;i<ArraySizeX;i++){
       for(j=0;j<ArraySizeY;j++)
         fprintf(fp, " %f", u_h[i*ArraySizeY+j]);
        fprintf(fp, "\n");
     } 
     fclose(fp);

     //cleanup
     clReleaseKernel(kernel);
     clReleaseKernel(kernel1);
     clReleaseProgram(program);
     clReleaseCommandQueue(cmdQueue);
     clReleaseMemObject(u_d);
     clReleaseMemObject(f_d);
     clReleaseContext(context);

     free(u_h);
     free(f_h);
     free(platforms);
     free(devices);
     
     return 0;
}


