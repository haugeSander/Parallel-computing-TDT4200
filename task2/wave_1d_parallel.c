#define _XOPEN_SOURCE 600
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

typedef int64_t int_t;
typedef double real_t;

int size, rank;
int_t local_N;
real_t *local_buffers[3] = {NULL, NULL, NULL};

const int_t
    N = 65536,
    max_iteration = 100000,
    snapshot_freq = 500;

const real_t
    c  = 1.0,
    dx = 1.0;
real_t dt;

real_t *global_buffer = NULL;

#define U_prv(i) local_buffers[0][(i)+1]
#define U(i)     local_buffers[1][(i)+1]
#define U_nxt(i) local_buffers[2][(i)+1]

#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

void domain_save(int_t step)
{
    if (rank == 0) {
        char filename[256];
        sprintf(filename, "data/%.5ld.dat", step);
        FILE *out = fopen(filename, "wb");
        fwrite(global_buffer, sizeof(real_t), N, out);
        fclose(out);
    }
}

void domain_initialize(void)
{
    local_N = N / size;
    if (rank == size - 1) {
        local_N += N % size;
    }

    for (int i = 0; i < 3; i++) {
        local_buffers[i] = malloc((local_N + 2) * sizeof(real_t));
    }

    for (int_t i = 0; i < local_N; i++) {
        real_t x = (rank * (N / size) + i) / (real_t)N;
        U_prv(i) = U(i) = cos(M_PI * x);
    }

    dt = dx / c;

    if (rank == 0) {
        global_buffer = malloc(N * sizeof(real_t));
    }
}

void domain_finalize(void)
{
    for (int i = 0; i < 3; i++) {
        free(local_buffers[i]);
    }
    if (rank == 0) {
        free(global_buffer);
    }
}

void move_buffer_window(void)
{
    real_t *temp = local_buffers[0];
    local_buffers[0] = local_buffers[1];
    local_buffers[1] = local_buffers[2];
    local_buffers[2] = temp;
}

void time_step(void)
{
    for (int_t i = 0; i < local_N; i++) {
        U_nxt(i) = -U_prv(i) + 2.0 * U(i)
                 + (dt * dt * c * c) / (dx * dx) * (U(i-1) + U(i+1) - 2.0 * U(i));
    }
}

void boundary_condition(void)
{
    if (rank == 0) {
        U(-1) = U(1);
    }
    if (rank == size - 1) {
        U(local_N) = U(local_N - 2);
    }
}

void border_exchange(void)
{
    MPI_Status status;
    if (rank > 0) {
        MPI_Send(&U(0), 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
        MPI_Recv(&U(-1), 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &status);
    }
    if (rank < size - 1) {
        MPI_Send(&U(local_N - 1), 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
        MPI_Recv(&U(local_N), 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &status);
    }
}

void send_data_to_root()
{
    MPI_Gather(&U(0), local_N, MPI_DOUBLE, global_buffer, local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void simulate(void)
{
    for (int_t iteration = 0; iteration <= max_iteration; iteration++)
    {
        if ((iteration % snapshot_freq == 0))
        {
            send_data_to_root();
            if (rank == 0) {
                domain_save(iteration / snapshot_freq);
            }
        }

        border_exchange();
        boundary_condition();
        time_step();

        move_buffer_window();
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    domain_initialize();

    double start_time = MPI_Wtime();

    simulate();

    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Total elapsed time: %f seconds\n", end_time - start_time);
    }
   
    domain_finalize();

    MPI_Finalize();
    return 0;
}