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
#include <mpi/mpi.h>
// END: T1a


// Option to change numerical precision.
typedef int64_t int_t;
typedef double real_t;


// TASK: T1b
// Declare variables each MPI process will need
// BEGIN: T1b
int rank, size, process_split, global_start;
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
real_t
    *buffers[3] = { NULL, NULL, NULL };


#define U_prv(i) buffers[0][(i)+1]
#define U(i)     buffers[1][(i)+1]
#define U_nxt(i) buffers[2][(i)+1]


// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)


// TASK: T8
// Save the present time step in a numbered file under 'data/'.
void domain_save ( int_t step )
{
// BEGIN: T8
    char filename[256];
    sprintf ( filename, "data/%.5ld.dat", step );
    FILE *out = fopen ( filename, "wb" );
    fwrite ( &U(0), sizeof(real_t), N, out );
    fclose ( out );
// END: T8
}


// TASK: T3
// Allocate space for each process' sub-grids
// Set up our three buffers, fill two with an initial cosine wave,
// and set the time step.
void domain_initialize ( void )
{
// BEGIN: T3
    int remainder = N % size;
    process_split = N / size;
    if (rank < remainder) {
        process_split++;
        global_start = rank * process_split;
    } else {
        global_start = rank * (N / size) + remainder;
    }    
    int buffer_size = process_split + 2;

    for (int i = 0; i < 3; i++) {
        buffers[i] = malloc(buffer_size * sizeof(real_t));
        if (buffers[i] == NULL) {
            fprintf(stderr, "Memory allocation failed for buffer %d\n", i);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    for (int_t i = 0; i < process_split; i++) {
        real_t x = (global_start + i) / (real_t)(N - 1);
        U_prv(i) = U(i) = cos(M_PI * x);
    }
// END: T3

    // Set the time step for 1D case.
    dt = dx / c;
    printf("Rank %d: Initialized with process_split = %d, global_start = %d\n", rank, process_split, global_start);
}


// Return the memory to the OS.
void domain_finalize ( void )
{
    free ( buffers[0] );
    free ( buffers[1] );
    free ( buffers[2] );
}


// Rotate the time step buffers.
void move_buffer_window ( void )
{
    real_t *temp = buffers[0];
    buffers[0] = buffers[1];
    buffers[1] = buffers[2];
    buffers[2] = temp;
}


// TASK: T4
// Derive step t+1 from steps t and t-1.
void time_step ( void )
{
// BEGIN: T4
    for ( int_t i=0; i<process_split; i++ )
    {
        real_t left = (i > 0 || rank > 0) ? U(i-1) : U(0);
        real_t right = (i < process_split-1 || rank < size-1) ? U(i+1) : U(process_split-1);

        U_nxt(i) = -U_prv(i) + 2.0*U(i)
                 + (dt*dt*c*c)/(dx*dx) * (left + right - 2.0 *U(i));
    }
// END: T4
}


// TASK: T6
// Neumann (reflective) boundary condition.
void boundary_condition ( void )
{
// BEGIN: T6
    if (rank == 0)
    {
        U(-1) = U(1);    
    }
    if (rank == size - 1)
    {
        U(N) = U(N-2);
    }
// END: T6
}


// TASK: T5
// Communicate the border between processes.
void border_exchange( void )
{
// BEGIN: T5
    MPI_Request requests[4];
    MPI_Status statuses[4];
    int request_count = 0;

    if (rank < size - 1) {
        MPI_Isend(&U(process_split-1), 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &requests[request_count++]);
        MPI_Irecv(&U(process_split), 1, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &requests[request_count++]);
    }
    if (rank > 0) {
        MPI_Isend(&U(0), 1, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, &requests[request_count++]);
        MPI_Irecv(&U(-1), 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &requests[request_count++]);
    }
    MPI_Waitall(request_count, requests, statuses);
// END: T5
}


// TASK: T7
// Every process needs to communicate its results
// to root and assemble it in the root buffer
void send_data_to_root()
{
// BEGIN: T7
    MPI_Status status;
    real_t *global_result = NULL;
    if (rank == 0) {
        global_result = (real_t *)malloc(N * sizeof(real_t));
        if (global_result == NULL) {
            fprintf(stderr, "Memory allocation failed for global_result\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Copy root process data
        memcpy(global_result, &U(0), process_split * sizeof(real_t));

        // Receive data from other processes
        for (int i = 1; i < size; i++) {
            int recv_count;
            int recv_start;
            
            MPI_Recv(&recv_count, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&recv_start, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(&global_result[recv_start], recv_count, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, &status);
        }        
        free(global_result);
    } else {
        // Send data to root
        MPI_Send(&process_split, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&global_start, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(&U(0), process_split, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }
// END: T7
}


// Main time integration.
void simulate( void )
{
    if (rank == 0) {
        printf("Entering simulate function\n");
        fflush(stdout);
    }

    for (int_t iteration = 0; iteration <= max_iteration; iteration++)
    {
        if (iteration % 1000 == 0 && rank == 0) {
            printf("Starting iteration %ld\n", iteration);
            fflush(stdout);
        }

        if (rank == 0 && (iteration % snapshot_freq == 0))
        {
            send_data_to_root();
            domain_save(iteration / snapshot_freq);
        }

        border_exchange();
        boundary_condition();
        time_step();
        move_buffer_window();

        if (iteration % 1000 == 0 && rank == 0) {
            printf("Completed iteration %ld\n", iteration);
            fflush(stdout);
        }
    }

    if (rank == 0) {
        printf("Exiting simulate function\n");
        fflush(stdout);
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
    
    domain_initialize();

// TASK: T2
// Time your code
// BEGIN: T2
    double start_time = MPI_Wtime();
    simulate();
    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Total elapsed time: %f seconds\n", end_time - start_time);
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
