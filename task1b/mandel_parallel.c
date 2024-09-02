#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi/mpi.h>
#include <time.h>

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

void calculate(int start_y, int end_y) {
	for(int j=start_y;j<end_y;j++) {
		for(int i=0;i<XSIZE;i++) {
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

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
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

    double start_time = get_time();

	/* Calculate the range in the y-axis such that we preserve the
	   aspect ratio */
	step=(xright-xleft)/XSIZE;
	yupper=ycenter+(step*YSIZE)/2;
	ylower=ycenter-(step*YSIZE)/2;
	
	int rank, size, message_Item;
	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (size < 2) {
		if (rank == 0) printf("This program must be run with 3 processes.\n");
    	MPI_Finalize();
    	return 1;
	}

    int rows_per_process = YSIZE / size;
    int remainder = YSIZE % size;
    int start_y = rank * rows_per_process + (rank < remainder ? rank : remainder);
    int end_y = start_y + rows_per_process + (rank < remainder ? 1 : 0);
    int local_rows = end_y - start_y;
	calculate(start_y, end_y);

    int *recvcounts = NULL;
    int *displs = NULL;
    if (rank == 0) {
        recvcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        for (int i = 0; i < size; i++) {
            recvcounts[i] = (rows_per_process + (i < remainder ? 1 : 0)) * XSIZE;
            displs[i] = (i * rows_per_process + (i < remainder ? i : remainder)) * XSIZE;
        }
    }
	// Gather all results to rank 0
    MPI_Gatherv(&pixel[start_y * XSIZE], local_rows * XSIZE, MPI_INT,
                pixel, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

	double end_time = get_time();
    if (rank == 0) {
        printf("Execution time: %f seconds\n", end_time - start_time);
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
