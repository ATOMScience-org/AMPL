import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import sys
from datetime import datetime
import atomsci.ddm.pipeline.parameter_parser as parse
import socket
import shutil
import subprocess
import time

def run_cmd(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    p.wait()
    return output

def run_command(shell_script, python_path, script_dir, params_str):
    params_str = params_str.replace('--descriptor_type mordred ', '--descriptor_type moe ')
    slurm_command = 'sbatch {0} {1} {2} "{3}"'.format(shell_script, python_path, script_dir, params_str)
    i = int(run_cmd('squeue | grep $(whoami) | wc -l').decode("utf-8"))
    print(i)
    while i >= 85:
        print("%d jobs in queue, sleeping" % i)
        time.sleep(60)
        i = int(run_cmd('squeue | grep $(whoami) | wc -l').decode("utf-8"))
    print(slurm_command)
    os.system(slurm_command)

def move_failed_slurm(output_dirs):
    for output_dir in output_dirs:
        files = [f for f in listdir(output_dir) if isfile(join(output_dir, f)) and 'slurm' in f]
        failed_output_dir = output_dir.replace('slurm_files', 'failed_slurm_files')
        if not os.path.isdir(failed_output_dir):
            os.mkdir(failed_output_dir)
        canceled_output_dir = output_dir.replace('slurm_files', 'canceled_slurm_files')
        if not os.path.isdir(canceled_output_dir):
            os.mkdir(canceled_output_dir)
        for filename in files:
            with open(os.path.join(output_dir, filename), 'r') as f:
                lines = [line.strip() for line in f.readlines()]
                try:
                    if 'Successfully inserted into the database' not in lines[-2]:
                        print(filename)
                        if 'CANCELLED' not in lines[-1]:
                            shutil.move(os.path.join(output_dir, filename), os.path.join(failed_output_dir, filename))
                        else:
                            shutil.move(os.path.join(output_dir, filename), os.path.join(canceled_output_dir, filename))
                except Exception as e:
                    print(e)
                    print(filename)


def rerun_failed(script_dir, python_path, output_dirs, result_dir):
    shell_script = os.path.join(result_dir, 'run.sh')
    hostname = ''.join(list(filter(lambda x: x.isalpha(), socket.gethostname())))
    for output_dir in output_dirs:
        with open(shell_script, 'w') as f:
            f.write("#!/bin/bash\n#SBATCH -A baasic\n#SBATCH -N 1\n#SBATCH -p partition={0}\n#SBATCH -t 24:00:00"
                    "\n#SBATCH -p pbatch\n#SBATCH --export=ALL\n#SBATCH -D {1}\n".format(hostname, output_dir))
            f.write('start=`date +%s`\necho $3\n$1 $2/pipeline/model_pipeline.py $3\nend=`date +%s`\n'
                  'runtime=$((end-start))\necho "runtime: " $runtime')
        files = [f for f in listdir(output_dir) if isfile(join(output_dir, f)) and 'slurm' in f]
        for filename in files:
            with open(os.path.join(output_dir, filename), 'r') as f:
                lines = [line.strip() for line in f.readlines()]
                try:
                    if 'Successfully inserted into the database' not in lines[-2]:
                        run_command(shell_script, python_path, script_dir, lines[0])
                except Exception as e:
                    print(e)
                    print(filename)

def get_timings(output_dirs):
    output_stats = []
    for output_dir in output_dirs:
        print(output_dir)
        files = [f for f in listdir(output_dir) if isfile(join(output_dir, f)) and 'slurm' in f]
        for filename in files:
            with open(os.path.join(output_dir, filename), 'r') as f:
                lines = [line.strip() for line in f.readlines()]
                if lines[-2] == 'Successfully inserted into the database.':
                    run_time = lines[-1].split(' ')[-1]
                    num_samples = ''
                    for line in lines:
                        if len(line.split(' ')) > 4 and line.split(' ')[3] == 'size:':
                            num_samples = line.split(' ')[4]
                            break
                    try:
                        tmp_dict = parse.wrapper(lines[0].split(' ')).__dict__
                        tmp_dict.update({'run_time': int(run_time), 'num_samples': int(num_samples), 'slurm_file': os.path.join(output_dir, filename)})
                    except Exception as e:
                        print(lines[0])
                        print(e)
                        continue
                    output_stats.append(tmp_dict)
                else:
                    print('{} did not work'.format(filename))
                    print(lines[-2])
    df = pd.DataFrame(output_stats)
    df.to_csv(os.path.join('/p/lustre1/minnich2/', 'timing_results_{}.csv'.format(datetime.now().strftime("%Y%m%d-%H%M%S"))), index=False)

def main():
    if sys.argv[1] == 'get_timings':
        get_timings(sys.argv[2:])
    elif sys.argv[1] == 'move_failed_slurm':
        move_failed_slurm(sys.argv[2:])
    else:
        rerun_failed(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])

if __name__=='__main__':
    if len(sys.argv) == 1:
        print("Need to provide script_dir, python path, and output_dir")
        sys.exit(1)
    main()
    sys.exit(0)
