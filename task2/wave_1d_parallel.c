#define _XOPEN_SOURCE 600
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>

// TASK: T1a
// Include the MPI headerfile
// BEGIN: T1a
#include <mpi.h>
// END: T1a


// Option to change numerical precision.
typedef int64_t int_t;
typedef double real_t;


// TASK: T1b
// Declare variables each MPI process will need
// BEGIN: T1b
int size, rank;
int_t local_N;
real_t *local_buffers[3] = {NULL, NULL, NULL};
// END: T1b


// Simulation parameters: size, step count, and how often to save the state.
const int_t
    N = 65536,
    max_iteration = 100000,
    snapshot_freq = 500;

// Wave equation parameters, time step is derived from the space step.
const real_t
    c  = 1.0,
    dx = 1.0;
real_t
    dt;

// Buffers for three time steps, indexed with 2 ghost points for the boundary.
real_t *global_buffer = NULL;

#define U_prv(i) local_buffers[0][(i)+1]
#define U(i)     local_buffers[1][(i)+1]
#define U_nxt(i) local_buffers[2][(i)+1]


// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)


// TASK: T8
// Save the present time step in a numbered file under 'data/'.
void domain_save ( int_t step )
{
// BEGIN: T8
    if (rank == 0) {
        char filename[256];
        sprintf ( filename, "data/%.5ld.dat", step );
        FILE *out = fopen ( filename, "wb" );
        // Saves the global_buffer which is process 0.
        fwrite ( global_buffer, sizeof(real_t), N, out );
        fclose ( out );
    }
// END: T8
}


// TASK: T3
// Allocate space for each process' sub-grids
// Set up our three buffers, fill two with an initial cosine wave,
// and set the time step.
void domain_initialize ( void )
{
// BEGIN: T3
    // Assign tasks
    local_N = N / size;
    if (rank == size - 1) {
        local_N += N % size;
    }
    // Allocate local buffers, based on local_N
    for (int i = 0; i < 3; i++) {
        local_buffers[i] = malloc((local_N + 2) * sizeof(real_t));
    }
    // Initialize the local buffer
    for (int_t i = 0; i < local_N; i++) {
        real_t x = (rank * (N / size) + i) / (real_t)N;
        U_prv(i) = U(i) = cos(M_PI * x);
    }
    // Allocate global buffer
    if (rank == 0) {
        global_buffer = malloc(N * sizeof(real_t));
    }
// END: T3

    // Set the time step for 1D case.
    dt = dx / c;
}


// Return the memory to the OS.
void domain_finalize ( void )
{
    for (int i = 0; i < 3; i++) {
        free( local_buffers[i] );
    }
    if (rank == 0) {
        free( global_buffer );
    }
}


// Rotate the time step buffers.
void move_buffer_window ( void )
{
    real_t *temp = local_buffers[0];
    local_buffers[0] = local_buffers[1];
    local_buffers[1] = local_buffers[2];
    local_buffers[2] = temp;
}


// TASK: T4
// Derive step t+1 from steps t and t-1.
void time_step ( void )
{
// BEGIN: T4
    // Find the U_nxt
    for (int_t i = 0; i < local_N; i++) {
        U_nxt(i) = -U_prv(i) + 2.0 * U(i)
                 + (dt * dt * c * c) / (dx * dx) * (U(i-1) + U(i+1) - 2.0 * U(i));
    }
// END: T4
}


// TASK: T6
// Neumann (reflective) boundary condition.
void boundary_condition ( void )
{
// BEGIN: T6
    // If process 0, set the first ghsot element to the second to first.
    if (rank == 0) {
        U(-1) = U(1);
    }
    // Else set last ghost to second next to last
    if (rank == size - 1) {
        U(local_N) = U(local_N - 2);
    }// END: T6
}


// TASK: T5
// Communicate the border between processes.
void border_exchange( void )
{
// BEGIN: T5
    MPI_Status status;
    if (rank > 0) {
        MPI_Send(&U(0), 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
        MPI_Recv(&U(-1), 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &status);
    }
    if (rank < size - 1) {
        MPI_Send(&U(local_N - 1), 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
        MPI_Recv(&U(local_N), 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &status);
    }
// END: T5
}


// TASK: T7
// Every process needs to communicate its results
// to root and assemble it in the root buffer
void send_data_to_root()
{
// BEGIN: T7
    MPI_Gather(&U(0), local_N, MPI_DOUBLE, global_buffer, local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
// END: T7
}


// Main time integration.
void simulate( void )
{
    // Go through each time step.
    for ( int_t iteration=0; iteration<=max_iteration; iteration++ )
    {
        if ( (iteration % snapshot_freq)==0 )
        {
            send_data_to_root();
            domain_save ( iteration / snapshot_freq );
        }

        // Derive step t+1 from steps t and t-1.
        border_exchange();
        boundary_condition();
        time_step();

        move_buffer_window();
    }
}


int main ( int argc, char **argv )
{
// TASK: T1c
// Initialise MPI
// BEGIN: T1c
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
// END: T1c
    
    struct timeval t_start, t_end;

    domain_initialize();

// TASK: T2
// Time your code
// BEGIN: T2
    gettimeofday ( &t_start, NULL );
    simulate();
    gettimeofday ( &t_end, NULL );

    if (rank == 0) {
        printf ( "Total elapsed time: %lf seconds\n",
           WALLTIME(t_end) - WALLTIME(t_start)
        );    
    }
// END: T2
   
    domain_finalize();

// TASK: T1d
// Finalise MPI
// BEGIN: T1d
    MPI_Finalize();
// END: T1d

    exit ( EXIT_SUCCESS );
}
