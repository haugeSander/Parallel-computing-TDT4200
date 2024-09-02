#!/bin/bash

echo "Running serial version..."
time_serial=$(./mandel_serial 0 | grep "Execution time" | awk '{print $3}')
echo "Serial time: $time_serial seconds"

for np in 6 8 10 12; do
    echo "Running parallel version with $np processes..."
    time_parallel=$(mpirun -np $np ./mandel_parallel 0 | grep "Execution time" | awk '{print $3}')
    speedup=$(echo "scale=2; $time_serial / $time_parallel" | bc)
    efficiency=$(echo "scale=2; $speedup / $np" | bc)
    echo "Parallel time: $time_parallel seconds"
    echo "Speedup: $speedup"
    echo "Efficiency: $efficiency"
    echo ""
done
