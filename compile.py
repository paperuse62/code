import os
import sys

source_code = sys.argv[1]
executable_name = sys.argv[2]

os.system(f"nvcc -O3 -arch=sm_80 -m64 -std=c++14 -rdc=true --compiler-options -fPIC -lm -ccbin g++ -Xcompiler -fopenmp -L/path/to/nvcomp/lib -I /path/to/nvcomp/include/ -I ./ {source_code} -lcuda -lcufile -lnvcomp_bitcomp -o {executable_name}")