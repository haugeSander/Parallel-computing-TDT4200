CC = mpicc
CFLAGS = -O3 -Wall -lm
TARGET = mandel_parallel
SRCS = mandel_parallel.c
NPROCS ?= 4

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^

run: $(TARGET)
	mpirun -np $(NPROCS) ./$(TARGET) 1

clean:
	rm -f $(TARGET) mandel2.bmp

.PHONY: all run clean