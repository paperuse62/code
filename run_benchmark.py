import os

bmlqsim_path = "."

algorithms = ["cat_state", "cc", "ising", "qft", "bv", "qsvm", "qaoa", "ghz"]

# translate qasm to c
for n in range(24, 35):
    for algorithm in algorithms:
        os.system(f"python {bmlqsim_path}/qasm_to_c.py -i {bmlqsim_path}/qasm_files/{algorithm}_n{n}.qasm -o ./cu_files/{algorithm}_n{n}.cu")

# compile c to executable
for n in range(24, 35):
    for algorithm in algorithms:
        os.system(f"python {bmlqsim_path}/compile.py ./cpp_files/{algorithm}_n{n}.cu ./executables/{algorithm}_n{n}")

# run executables
for n in range(24, 35):
    for algorithm in algorithms:
        os.system(f"./executables/{algorithm}_n{n} > ./results/{algorithm}_n{n}.txt")