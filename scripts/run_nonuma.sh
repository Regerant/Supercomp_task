THREADS=(1 2 4 8 16 28 32)

echo "-------"
echo "  GCC  "
echo "-------"
gcc -fopenmp task_16f.c -o taskf_gcc_nonuma

for t in "${THREADS[@]}"; do
  echo "----------"
  echo "Threads = $t"
  export OMP_NUM_THREADS=$t
  # Убираем привязку
  unset OMP_PROC_BIND
  unset OMP_PLACES
  ./taskf_gcc_nonuma
done

echo
echo "--------"
echo "  ICX   "
echo "--------"
icx -fiopenmp task_16f.c -o taskf_icx_nonuma

for t in "${THREADS[@]}"; do
  echo "----------"
  echo "Threads = $t"
  export OMP_NUM_THREADS=$t
  unset OMP_PROC_BIND
  unset OMP_PLACES
  ./taskf_icx_nonuma
done
