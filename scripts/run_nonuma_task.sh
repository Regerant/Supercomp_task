THREADS=(1 2 4 8 16 28 32)

echo "-------"
echo "  GCC  "
echo "-------"
gcc -fopenmp task_16t.c -o taskt_gcc_nonuma

for t in "${THREADS[@]}"; do
  echo "----------"
  echo "Threads = $t"
  export OMP_NUM_THREADS=$t
  unset OMP_PROC_BIND
  unset OMP_PLACES
  ./taskt_gcc_nonuma
done

echo
echo "--------"
echo "  ICX   "
echo "--------"
icx -fiopenmp task_16t.c -o taskt_icx_nonuma

for t in "${THREADS[@]}"; do
  echo "----------"
  echo "Threads = $t"
  export OMP_NUM_THREADS=$t
  unset OMP_PROC_BIND
  unset OMP_PLACES
  ./taskt_icx_nonuma
done
