import subprocess
import sys

BINARY_PATH='./build/ppp_proj01'
GENERATOR_PATH='./build/data_generator'

domain_sizes = [16, 32, 64]
nps = [1, 2, 4, 8, 16, 32]
iterations = [0]
modes = [1, 2]
num = 0
decomps = ['', '-g']

with open('output.txt', 'w') as f:
    for domain_size in domain_sizes:
        for np in nps:
            if np >= domain_size:
                continue
            for m in modes:
                if np == 1 and m == 2:
                    continue
                for decomp in decomps:
                    if np == 1 and decomp == '-g':
                        continue
                    for it in iterations:
                        num += 1
                        cmd = f"{GENERATOR_PATH} -n {domain_size} && mpirun --oversubscribe -np {np} {BINARY_PATH} -m {m} {decomp} -n {it} -i ppp_input_data.h5 -v"
                        f.write(cmd)
                        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        print(result.stderr.decode())
                        # Check the return code
                        if 'FAILED' in result.stdout.decode():
                            print(f"Command failed on:")
                            print(f"{cmd}")
                            f.write(result.stdout.decode())
                        else:
                            print(f'OK {num}')
