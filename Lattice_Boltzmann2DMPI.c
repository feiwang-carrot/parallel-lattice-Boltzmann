#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <math.h>
#include <string.h>

int Rows, Cols;
double ***f, **rho;
double w1 = 4.0/9.0, w2 = 1.0/9.0, w3 = 1.0/36.0;
double Amp = 0.001, Amp2 = 0.1, Width = 10, omega = 1;

void init(int rows, int cols, int idr,int idc, int r1, int r2){
int x, y, incr,incc;
double n, ux, uy, uxx, uyy, uxy, usq;
printf("index %d %d initialize \n", idr,idc);
fflush(stdout);
// add increment to make sure the initial condition is 
// consistent on different blocks;
if(idr < r1) incr = idr;
else incr = r1;
if(idc < r2) incc = idc;
else incc = r2;
   for(x = 0;x<rows;x++){
	for(y =0;y<cols;y++){
	// define the macroscopic properties of the initial condition.
	n =  1 + Amp2*exp(-(pow(x+incr+idr*(rows-2)-Rows/2,2)+pow(y+incc+idc*(cols-2)-Cols/2,2))/Width);
        ux = 0;
        uy = 0;		
	// intialize f to be the local equilibrium values	
	uxx = ux*ux;
	uyy = uy*uy;
	uxy = 2*ux*uy;
	usq = uxx+ uyy;
	  
        f[x][y][0] = w1*n*(1-1.5*usq);
        f[x][y][1] = w2*n*(1+3*ux+4.5*uxx-1.5*usq);
	f[x][y][2] = w2*n*(1-3*ux+4.5*uxx-1.5*usq);
	f[x][y][3] = w2*n*(1+3*uy+4.5*uyy-1.5*usq);
        f[x][y][4] = w2*n*(1-3*uy+4.5*uyy-1.5*usq); 
        f[x][y][5] = w3*n*(1+3*(ux+uy)+4.5*(uxx+uxy+uyy)-1.5*usq);
        f[x][y][6] = w3*n*(1+3*(-ux+uy)+4.5*(uxx-uxy+uyy)-1.5*usq);
	f[x][y][7] = w3*n*(1+3*(-ux-uy)+4.5*(uxx+uxy+uyy)-1.5*usq);
	f[x][y][8] = w3*n*(1+3*(ux-uy)+4.5*(uxx-uxy+uyy)-1.5*usq);
      //    printf("%f ",n);
	}//  printf("\n");
    }	
}

void DenRho(int rows,int cols){
  int x,y;
    for (x=0;x<rows;x++)
      for (y=0;y<cols;y++){
	rho[x][y]=f[x][y][0]+f[x][y][1]+f[x][y][2]+f[x][y][3]+f[x][y][4]
	  +f[x][y][5]+f[x][y][6]+f[x][y][7]+f[x][y][8];
      }
    /* for (x=0;x<xdim;x++)
      for (y=0;y<ydim;y++){
	n=f[x][y][0]+f[x][y][1]+f[x][y][2]+f[x][y][3]+f[x][y][4]
	  +f[x][y][5]+f[x][y][6]+f[x][y][7]+f[x][y][8];
	u[x][y][0]=f[x][y][1]-f[x][y][2]+f[x][y][5]-f[x][y][6]-f[x][y][7]+f[x][y][8];
	u[x][y][1]=f[x][y][3]-f[x][y][4]+f[x][y][5]+f[x][y][6]-f[x][y][7]-f[x][y][8];
	u[x][y][0]/=n;
	u[x][y][1]/=n;
      }*/
}
void iteration(int rows, int cols){
  int x,y;
  register double n,ux,uy,uxx,uyy,uxy,usq,Fx,Fy,Fxx,Fyy,Fxy,Fsq;
  // first we perform the collision step 
  for (x=0;x<rows;x++){
    for (y=0;y<cols;y++){
      n=f[x][y][0]+f[x][y][1]+f[x][y][2]+f[x][y][3]+f[x][y][4]+f[x][y][5]+f[x][y][6]+f[x][y][7]+f[x][y][8];
      ux=f[x][y][1]-f[x][y][2]+f[x][y][5]-f[x][y][6]-f[x][y][7]+f[x][y][8];
      uy=f[x][y][3]-f[x][y][4]+f[x][y][5]+f[x][y][6]-f[x][y][7]-f[x][y][8];
      ux/=n;
      uy/=n;
      uxx=ux*ux;
      uyy=uy*uy;
      uxy=2*ux*uy;
      usq=uxx+uyy;
      // implement the forcing terms and perform stream step
      Fx=0;//Amp*sin(y*2*M_PI/cols);
      Fy=0;
      Fxx=2*n*Fx*ux;
      Fyy=2*n*Fy*uy;
      Fxy=2*n*(Fx*uy+Fy*ux);
      Fsq=Fxx+Fyy;
      Fx*=n;
      Fy*=n;
      f[x][y][0]+=omega*(w1*n*(1-1.5*usq)-f[x][y][0])-w1*1.5*Fsq;
      f[x][y][1]+=omega*(w2*n*(1+3*ux+4.5*uxx -1.5*usq)-f[x][y][1])+w2*(3*Fx+4.5*Fxx-1.5*Fsq);
      f[x][y][2]+=omega*(w2*n*(1-3*ux+4.5*uxx -1.5*usq)-f[x][y][2])+w2*(-3*Fx+4.5*Fxx-1.5*Fsq);
      f[x][y][3]+=omega*(w2*n*(1+3*uy+4.5*uyy -1.5*usq)-f[x][y][3])+w2*(3*Fy+4.5*Fyy-1.5*Fsq);
      f[x][y][4]+=omega*(w2*n*(1-3*uy+4.5*uyy -1.5*usq)-f[x][y][4])+w2*(-3*Fy+4.5*Fyy-1.5*Fsq);
      f[x][y][5]+=omega*(w3*n*(1+3*(ux+uy)+4.5*(uxx+uxy+uyy)-1.5*usq)-f[x][y][5])+w3*(3*(Fx+Fy)+4.5*(Fxx+Fxy+Fyy)-1.5*Fsq);
      f[x][y][6]+=omega*(w3*n*(1+3*(-ux+uy)+4.5*(uxx-uxy+uyy)-1.5*usq)-f[x][y][6])+w3*(3*(-Fx+Fy)+4.5*(Fxx-Fxy+Fyy)-1.5*Fsq);
      f[x][y][7]+=omega*(w3*n*(1+3*(-ux-uy)+4.5*(uxx+uxy+uyy)-1.5*usq)-f[x][y][7])+w3*(3*(-Fx-Fy)+4.5*(Fxx+Fxy+Fyy)-1.5*Fsq);
      f[x][y][8]+=omega*(w3*n*(1+3*(ux-uy)+4.5*(uxx-uxy+uyy)-1.5*usq)-f[x][y][8])+w3*(3*(Fx-Fy)+4.5*(Fxx-Fxy+Fyy)-1.5*Fsq);
      
     }
     }
     // Now we can move the densities along the lattice.
    for(x=0;x<rows;x++) 
	 for(y =0;y<cols;y++){
	  f[x][y][2] = f[(x+1)%rows][y][2];
	  f[rows-1-x][y][1] = f[(rows-2-x+rows)%rows][y][1];
	  f[x][y][4] = f[x][(y+1)%cols][4];
	  f[x][cols-1-y][3] = f[x][(cols-2-y+cols)%cols][3];
          f[x][y][7] = f[(x+1)%rows][(y+1)%cols][7];
          f[rows-1-x][cols-1-y][5] = f[(rows-2-x+rows)%rows][(cols-2-y+cols)%cols][5];	  
	  f[x][cols-1-y][6] = f[(x+1)%rows][(cols-2-y+cols)%cols][6];
	  f[rows-1-x][y][8] = f[(rows-2-x+rows)%rows][(y+1)%cols][8];
	 }
} 

 main(int argc, char** argv){
  int i,j,k,T,nid,noprocs,M,N,remainder1,remainder2,size1,size2;
  double *elements, elements1,t1,t2,tc1,tc2;  
  int dims[2];
  int coord[2];
  int isperiodic[2];
  char str[20];
  FILE *fp;
  MPI_Comm comm2d;
  MPI_Datatype stride;
 //  dims[0] = 3;
 //  dims[1] = 3;
  isperiodic[0] = isperiodic[1]= 0;
  MPI_Status status;
  MPI_Request req_send10, req_send20, req_send30, req_send40; //
  MPI_Request req_recv10, req_recv20, req_recv30, req_recv40; //
  MPI_Init(&argc, &argv);
  fp = fopen ("Input.txt", "rt");
  fscanf(fp,"%d %d %d",&Rows,&dims[0],&dims[1]);
  MPI_Cart_create(MPI_COMM_WORLD,2,dims,isperiodic,1,&comm2d);
  MPI_Comm_rank(comm2d, &nid);
  MPI_Comm_size(comm2d, &noprocs);
  MPI_Cart_coords(comm2d,nid,2,coord);

  t2 = 0;
  t1 = MPI_Wtime();
  T = 100;
 // Rows = 31;
  Cols = Rows;
  
  //break compute grid among processors
  N = Rows -1;
  M = Cols -1;
  remainder1 = (N-1) % dims[0];
  remainder2 = (M-1) % dims[1];
  size1 = (N-1-remainder1)/dims[0];
  size2 = (M-1-remainder2)/dims[1];
  if(coord[0] < remainder1)
    size1 = size1 + 2;
  else
    size1 = size1 + 1;
  if(coord[1] < remainder2)
    size2 = size2 + 2;
  else 
    size2 = size2 + 1;
  // allocating cotiguous 3d array 
  f = (double ***) calloc(size1+1,sizeof(double **));
  elements = (double *) calloc(9*(size2+1)*(size1+1),sizeof(double));
  for(i=0;i<size1+1;i++){
     f[i]=(double **) calloc(size2+1, sizeof(double *));
     for(j=0;j<size2+1;j++){
       f[i][j] = elements+ (i*(size2+1)*9)+(j*9);
     }
  }
  rho = (double **) calloc(size1+1,sizeof(double *));
  rho[0] = (double *) calloc((size2+1)*(size1+1),sizeof(double));
  for(i=1;i<size1+1;i++)
     rho[i] =rho[i-1]+ size2+1;
 // printf("nid %d has size %d \n",nid, size-1);
  //  Access array elements 
 
  init(size1+1,size2+1,coord[0],coord[1],remainder1,remainder2); // initallize the density and particle distribution mesh
  // fp = fopen ("iteration.txt", "rt");  
  // fscanf(fp,"%d",&T);
   tc2 = 0;
  MPI_Type_vector((size1+1),9,(size2+1)*9,MPI_DOUBLE_PRECISION,&stride);
  MPI_Type_commit(&stride);
  for(i=0;i<T;i++){
      tc1 = MPI_Wtime(); 
     req_send10 = req_send20= req_recv10 = req_recv20 = MPI_REQUEST_NULL;
     MPI_Isend(f[size1-1][0],(size2+1)*9,MPI_DOUBLE,((coord[0]+1)%dims[0])*dims[1]+coord[1],10,comm2d,&req_send10);
     MPI_Irecv(f[0][0],(size2+1)*9,MPI_DOUBLE,((coord[0]-1+dims[0])%dims[0])*dims[1]+coord[1],10,comm2d,&req_recv10);
     MPI_Isend(f[1][0], (size2+1)*9, MPI_DOUBLE,((coord[0]-1+dims[0])%dims[0])*dims[1]+coord[1],20,comm2d,&req_send20);
     MPI_Irecv(f[size1][0],(size2+1)*9, MPI_DOUBLE,((coord[0]+1)%dims[0])*dims[1]+coord[1], 20, comm2d,&req_recv20);
       
     MPI_Wait(&req_recv10,&status);
     MPI_Wait(&req_recv20,&status);
     req_send30 = req_send40= req_recv30 = req_recv40 = MPI_REQUEST_NULL;
     MPI_Isend(f[0][size2-1],1,stride,dims[1]*coord[0]+(coord[1]+1)%dims[1],30,comm2d,&req_send30);
     MPI_Irecv(f[0][0],1,stride,dims[1]*coord[0]+(coord[1]-1+dims[1])%dims[1],30,comm2d,&req_recv30);
     MPI_Isend(f[0][1],1,stride,dims[1]*coord[0]+(coord[1]-1+dims[1])%dims[1],40,comm2d,&req_send40);
     MPI_Irecv(f[0][size2],1,stride,dims[1]*coord[0]+(coord[1]+1)%dims[1],40,comm2d,&req_recv40);
     
     MPI_Wait(&req_recv30,&status);
     MPI_Wait(&req_recv40,&status);
     tc2 =tc2+ MPI_Wtime()-tc1;  
   //periodic boundary conditions
    /*
      for(j=0;j<size+1;j++){
	    f[j][0][3] = f[j][Cols-2][3];
	    f[j][0][5] = f[j][Cols-2][5];
	    f[j][0][6] = f[j][Cols-2][6];
            f[j][Cols-1][4] = f[j][1][4];
	    f[j][Cols-1][7] = f[j][1][7];
	    f[j][Cols-1][8] = f[j][1][8]; 
     }*/
       iteration(size1+1,size2+1);  
    }
     DenRho(size1+1,size2+1);
       t2 =t2+ MPI_Wtime()-t1;
      printf("size1 %d, size2 %d \n",size1,size2);
      fflush(stdout); 
     if(nid==0){
       sprintf(str,"Timing.txt");
       fp = fopen(str,"wt");
       fprintf(fp,"%6.4f %6.4f",t2,tc2);
    }
    printf("Total Time is %f, communication time is %f\n", t2,tc2);
  
      /*
      printf("\n");
      if(1 ==1){
      for(i=0;i<size1+1;i++){
        for(j=0;j<size2+1;j++){
        printf("%f ", rho[i][j]);    
        }
         printf("\n");}}*/
/* Output result */
  
 /*  sprintf(str,"2Solution%d.txt",nid);
   fp = fopen(str,"wt");
   for(i = 1; i < size1; i++){
   for(j = 1; j < size2; j++)
   fprintf(fp,"%6.4f ",rho[i][j]);
   fprintf(fp,"\n");
   }
   fclose(fp);*/
 
   MPI_Finalize();
}
