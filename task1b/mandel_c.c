#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi/mpi.h>

#define XSIZE 2560
#define YSIZE 2048

#define MAXITER 255

double xleft=-2.01;
double xright=1;
double yupper,ylower;
double ycenter=1e-6;
double step;

int pixel[XSIZE*YSIZE];

#define PIXEL(i,j) ((i)+(j)*XSIZE)

typedef struct {
	double real,imag;
} complex_t;

void calculate(int start_x, int end_x) {
	for(int i=start_x;i<end_x;i++) {
		for(int j=0;j<YSIZE;j++) {
			/* Calculate the number of iterations until divergence for each pixel.
			   If divergence never happens, return MAXITER */
			complex_t c,z,temp;
			int iter=0;
			c.real = (xleft + step*i);
			c.imag = (ylower + step*j);
			z = c;
			while(z.real*z.real + z.imag*z.imag < 4) {
				temp.real = z.real*z.real - z.imag*z.imag + c.real;
				temp.imag = 2*z.real*z.imag + c.imag;
				z = temp;
				if(++iter==MAXITER) break;
			}
			pixel[PIXEL(i,j)]=iter;
		}
	}
}

typedef unsigned char uchar;

/* save 24-bits bmp file, buffer must be in bmp format: upside-down */
void savebmp(char *name,uchar *buffer,int x,int y) {
	FILE *f=fopen(name,"wb");
	if(!f) {
		printf("Error writing image to disk.\n");
		return;
	}
	unsigned int size=x*y*3+54;
	uchar header[54]={'B','M',size&255,(size>>8)&255,(size>>16)&255,size>>24,0,
		0,0,0,54,0,0,0,40,0,0,0,x&255,x>>8,0,0,y&255,y>>8,0,0,1,0,24,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	fwrite(header,1,54,f);
	fwrite(buffer,1,XSIZE*YSIZE*3,f);
	fclose(f);
}

/* given iteration number, set a colour */
void fancycolour(uchar *p,int iter) {
	if(iter==MAXITER);
	else if(iter<8) { p[0]=128+iter*16; p[1]=p[2]=0; }
	else if(iter<24) { p[0]=255; p[1]=p[2]=(iter-8)*16; }
	else if(iter<160) { p[0]=p[1]=255-(iter-24)*2; p[2]=255; }
	else { p[0]=p[1]=(iter-160)*2; p[2]=255-(iter-160)*2; }
}

int main(int argc,char **argv) {
	if(argc==1) {
		puts("Usage: MANDEL n");
		puts("n decides whether image should be written to disk (1=yes, 0=no)");
		return 0;
	}
	/* Calculate the range in the y-axis such that we preserve the
	   aspect ratio */
	step=(xright-xleft)/XSIZE;
	yupper=ycenter+(step*YSIZE)/2;
	ylower=ycenter-(step*YSIZE)/2;
	
	int rank, size, message_Item;
	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (size != 3) {
		if (rank == 0) printf("This program must be run with 3 processes.\n");
    	MPI_Finalize();
    	return 1;
	}

	int columns_per_process = XSIZE / 3;
	int start_x = rank * columns_per_process;
	int end_x = (rank == 2) ? XSIZE : (rank + 1) * columns_per_process;

	calculate(start_x, end_x);

	if (rank != 0) {
	    MPI_Send(&pixel[start_x], columns_per_process * YSIZE, MPI_INT, 0, 0, MPI_COMM_WORLD);
	} else {
	    for (int i = 1; i < 3; i++) {
	        MPI_Recv(&pixel[i * columns_per_process], columns_per_process * YSIZE, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    }
	}

	if(rank == 0 && strtol(argv[1],NULL,10)!=0) {
		/* create nice image from iteration counts. take care to create it upside
		   down (bmp format) */
		unsigned char *buffer=calloc(XSIZE*YSIZE*3,1);
		for(int i=0;i<XSIZE;i++) {
			for(int j=0;j<YSIZE;j++) {
				int p=((YSIZE-j-1)*XSIZE+i)*3;
				fancycolour(buffer+p,pixel[PIXEL(i,j)]);
			}
		}
		/* write image to disk */
		savebmp("mandel2.bmp",buffer,XSIZE,YSIZE);
		free(buffer);
	}
	MPI_Finalize();
	return 0;
}
