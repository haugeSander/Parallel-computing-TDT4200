//#define _XOPEN_SOURCE 600
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>

// TASK: T1
// Include the cooperative groups library
// BEGIN: T1
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
// END: T1


// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

// Option to change numerical precision
typedef int64_t int_t;
typedef double real_t;

// TASK: T1b
// Variables needed for implementation
// BEGIN: T1b

// Simulation parameters: size, step count, and how often to save the state
int_t
    N = 128,
    M = 128,
    max_iteration = 100000,//0,
    snapshot_freq = 1000;

// Wave equation parameters, time step is derived from the space step
const real_t
    c  = 1.0,
    dx = 1.0,
    dy = 1.0;
real_t
    dt;

// Buffers for three time steps, indexed with 2 ghost points for the boundary
real_t
    *buffers[3] = { NULL, NULL, NULL };

#define U_prv(i,j) buffers[0][((i)+1)*(N+2)+(j)+1]
#define U(i,j)     buffers[1][((i)+1)*(N+2)+(j)+1]
#define U_nxt(i,j) buffers[2][((i)+1)*(N+2)+(j)+1]


// Divide the problem into blocks of BLOCKX x BLOCKY threads
#define BLOCKY 16
#define BLOCKX 16

// Global CUDA prop information
cudaDeviceProp prop;

// Device-side variables
real_t *d_prv = NULL;   // Previous time step on device
real_t *d_current = NULL;  // Current time step on device
real_t *d_nxt = NULL;    // Next time step on device
// END: T1b

#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Rotate the time step buffers.
void move_buffer_window ( void )
{
    real_t *temp = buffers[0];
    buffers[0] = buffers[1];
    buffers[1] = buffers[2];
    buffers[2] = temp;

    // Rotate pointers for next iteration
    real_t *d_temp = d_prv;
    d_prv = d_current;
    d_current = d_nxt;
    d_nxt = d_temp;
}

// Save the present time step in a numbered file under 'data/'
void domain_save ( int_t step )
{
    char filename[256];
    sprintf ( filename, "data/%.5ld.dat", step );
    FILE *out = fopen ( filename, "wb" );
    for ( int_t i=0; i<M; i++ )
    {
        fwrite ( &U(i,0), sizeof(real_t), N, out );
    }
    fclose ( out );
}


// TASK: T4
// Get rid of all the memory allocations
void domain_finalize ( void )
{
// BEGIN: T4
    // Free memory on host
    free ( buffers[0] );
    free ( buffers[1] );
    free ( buffers[2] );

    // Free memory on device
    cudaFree(d_prv);
    cudaFree(d_current);
    cudaFree(d_nxt);
// END: T4
}


// TASK: T6
// Neumann (reflective) boundary condition
// BEGIN: T6
__global__ void device_boundary_condition(real_t* d_current, int N, int M) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Handle boundary conditions
    if (i < M) {
        if (j == 0) {
            d_current[(i+1)*(N+2)] = d_current[(i+1)*(N+2)+2];
        }
        if (j == N-1) {
            d_current[(i+1)*(N+2)+N+1] = d_current[(i+1)*(N+2)+N-1];
        }
    }
    if (j < N) {
        if (i == 0) {
            d_current[(j+1)] = d_current[2*(N+2)+(j+1)];
        }
        if (i == M-1) {
            d_current[(M+1)*(N+2)+(j+1)] = d_current[(M-1)*(N+2)+(j+1)];
        }
    }
}
// END: T6


// TASK: T5
// Integration formula
// BEGIN; T5
__global__ void device_time_step ( real_t* d_prv, real_t* d_current, real_t* d_nxt, int N, int M, real_t dt, real_t dx, real_t dy, real_t c )
{
    // Calculate global thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Synchronize the grid
    //cg::this_grid().sync(); 
    //device_boundary_condition(d_current, N, M);

    // Check if thread is within bounds
    if (i < M && j < N) {
        // Adjust indices for ghost cells
        int idx = (i+1)*(N+2) + (j+1);
        int idx_up = ((i-1)+1)*(N+2) + (j+1);
        int idx_down = ((i+1)+1)*(N+2) + (j+1);
        int idx_left = (i+1)*(N+2) + (j-1+1);
        int idx_right = (i+1)*(N+2) + (j+1+1);

        d_nxt[idx] = -d_prv[idx] + 2.0*d_current[idx]
                   + (dt*dt*c*c)/(dx*dy) * (
                      d_current[idx_up] + d_current[idx_down] + 
                      d_current[idx_left] + d_current[idx_right] - 
                      4.0*d_current[idx]
                   );
    }
    cg::this_grid().sync(); 
}
// END: T5


// TASK: T7
// Main time integration.
void simulate( void )
{
// BEGIN: T7
    // Calculate the dimensions of the grid of blocks.
    dim3 block(BLOCKX, BLOCKY);
    // Even if XSIZE or YSIZE is not evenly divisible by BLOCKX or BLOCKY it will divide it as evenly as possible.
    dim3 grid((M + BLOCKX - 1) / BLOCKX, 
             ( N + BLOCKY - 1) / BLOCKY);

    // Debug prints
    printf("Grid dimensions: %dx%d\n", grid.x, grid.y);
    printf("Block dimensions: %dx%d\n", block.x, block.y);
    printf("Domain dimensions: %ldx%ld\n", M, N);
    printf("dt=%f, dx=%f, dy=%f, c=%f\n", dt, dx, dy, c);

    // Go through each time step
    for ( int_t iteration=0; iteration<=max_iteration; iteration++ ) {
        if ( (iteration % snapshot_freq)==0 )
        {
            // Make sure all kernels have completed before copying data
            cudaDeviceSynchronize();
            // Copy current state back to host for saving
            cudaMemcpy(buffers[1], d_current, (M + 2) * (N + 2) * sizeof(real_t), cudaMemcpyDeviceToHost);
            // Save the current state
            domain_save ( iteration / snapshot_freq );
                    
            // Rotate the time step buffers
            move_buffer_window();
        }
        
        device_boundary_condition<<<grid, block>>>(d_current, N, M);
        // Synchronize the findings
        cudaDeviceSynchronize();

        // Compute next time step
        device_time_step<<<grid, block>>>(d_prv, d_current, d_nxt, N, M, dt, dx, dy, c);
        // Synchronize the findings
        cudaDeviceSynchronize();
    }
// END: T7
}


// TASK: T8
// GPU occupancy
void occupancy( void )
{
// BEGIN: T8
    int maxActiveBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks,
        device_time_step,
        BLOCKX * BLOCKY,  // threads per block
        0     // shared memory size
    );
    int activeWarps = maxActiveBlocks * (BLOCKX * BLOCKY) / BLOCKX;  // 32 threads per warp
    int maxWarps = prop.maxThreadsPerMultiProcessor / BLOCKX;

    float occupancyRate = (float) activeWarps / maxWarps;
    printf("Theoretical occupancy: %.2f%%\n", occupancyRate * 100);
// END: T8
}


// TASK: T2
// Make sure at least one CUDA-capable device exists
static bool init_cuda()
{
// BEGIN: T2
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        printf("Error: No CUDA-compatible GPU device found!\n");
        return false;
    }

    // Use first available device
    cudaSetDevice(0);
    
    // Get and print device properties
    cudaGetDeviceProperties(&prop, 0);
    
    printf("Device name: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max thread dimensions: (%d, %d, %d)\n", 
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Number of multiprocessors: %d\n", prop.multiProcessorCount);
    
    return true;
// END: T2
}


// TASK: T3
// Set up our three buffers, and fill two with an initial perturbation
void domain_initialize ( void )
{
// BEGIN: T3
    // Check if CUDA exists
    if (!init_cuda())
    {
        fprintf(stderr, "CUDA initialization failed\n");
        exit( EXIT_FAILURE );
    }
    // Calculate the necessary memory that must be allocated
    size_t size = (M + 2) * (N + 2) * sizeof(real_t);
    
    // Initialize ALL host buffers using calloc
    buffers[0] = (real_t*)calloc((M + 2) * (N + 2), sizeof(real_t));
    buffers[1] = (real_t*)calloc((M + 2) * (N + 2), sizeof(real_t));
    buffers[2] = (real_t*)calloc((M + 2) * (N + 2), sizeof(real_t));
  
    // Allocate 3 buffers to the device
    cudaMalloc((void**) &d_prv, size);
    cudaMalloc((void**) &d_current, size);
    cudaMalloc((void**) &d_nxt, size);

    for ( int_t i=0; i<M; i++ )
    {
        for ( int_t j=0; j<N; j++ )
        {
            // Calculate delta (radial distance) adjusted for M x N grid
            real_t delta = sqrt ( ((i - M/2.0) * (i - M/2.0)) / (real_t)M +
                                  ((j - N/2.0) * (j - N/2.0)) / (real_t)N );
            U_prv(i,j) = U(i,j) = exp ( -4.0*delta*delta );
        }
    }
    // Copy initialized onto device
    cudaMemcpy(d_prv, buffers[0], size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_current, buffers[1], size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_nxt, buffers[2], size, cudaMemcpyHostToDevice);

    // Set the time step for 2D case
    dt = dx*dy / (c * sqrt (dx*dx+dy*dy));
// END: T3
}


int main ( void )
{
    // Set up the initial state of the domain
    domain_initialize();

    struct timeval t_start, t_end;

    gettimeofday ( &t_start, NULL );
    simulate();
    gettimeofday ( &t_end, NULL );

    printf ( "Total elapsed time: %lf seconds\n",
        WALLTIME(t_end) - WALLTIME(t_start)
    );

    occupancy();

    // Clean up and shut down
    domain_finalize();
    exit ( EXIT_SUCCESS );
}
