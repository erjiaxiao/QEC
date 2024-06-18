import time
import subprocess


def update_settings_file(file_path, PER):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Modify the training parameters
    new_lines = []
    for line in lines:
        if line.strip().startswith('#define P_TRAIN_3'):
            new_lines.append(f'#define P_TRAIN_3 {PER:.5f}\n')
        elif line.strip().startswith('#define P_TRAIN_5'):
            new_lines.append(f'#define P_TRAIN_5 {PER:.5f}\n')
        elif line.strip().startswith('#define P_TRAIN_7'):
            new_lines.append(f'#define P_TRAIN_7 {PER:.5f}\n')
        elif line.strip().startswith('#define P_TRAIN_9'):
            new_lines.append(f'#define P_TRAIN_9 {PER:.5f}\n')
        else:
            new_lines.append(line)
            
    with open(file_path, 'w') as file:
        file.writelines(new_lines)


PER = 0.1
DISTANCE = 3
setting_path = 'gpu_code/cuda/include/settings.h'

start_time = time.time()

update_settings_file(setting_path, PER)

compile_result = subprocess.run("/usr/local/cuda-11.7/bin/nvcc -I gpu_code/cuda/include/ -g -G gpu_code/cuda/src/main.cu -o executest", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
# print("Compile Output:", compile_result.stdout)
# print("Compile Errors:", compile_result.stderr)

execute_result = subprocess.run(f"./executest -d {DISTANCE} -t 2 -s 0 -q 0 -r 1 -1 256 -2 64", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
# print("Debug Output:", execute_result.stdout)
# print("Debug Errors:", execute_result.stderr)

print(execute_result.stdout)

end_time = time.time()
elapsed_time_hours = (end_time - start_time) / 3600
print(f"total time taken: {elapsed_time_hours:.3f} hours")