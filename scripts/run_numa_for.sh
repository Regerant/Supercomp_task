THREADS=(1 2 4 8 16 28 32)


echo "-----------------------------------------------------"
echo "  GCC + NUMA (OMP_PROC_BIND=spread, OMP_PLACES=cores)"
echo "-----------------------------------------------------"
gcc -fopenmp task_16f.c -o taskf_gcc_numa

for t in "${THREADS[@]}"; do
  echo "----------"
  echo "Threads = $t"
  export OMP_NUM_THREADS=$t
  export OMP_PROC_BIND=spread
  export OMP_PLACES=cores
  ./taskf_gcc_numa
done

echo
echo "-----------------------------------------------------"
echo "  ICX + NUMA (OMP_PROC_BIND=spread, OMP_PLACES=cores)"
echo "-----------------------------------------------------"
icx -fiopenmp task_16f.c -o taskf_icx_numa

for t in "${THREADS[@]}"; do
  echo "----------"
  echo "Threads = $t"
  export OMP_NUM_THREADS=$t
  export OMP_PROC_BIND=spread
  export OMP_PLACES=cores
  ./taskf_icx_numa
done