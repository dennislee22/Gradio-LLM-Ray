import subprocess, os, time
import cml.workers_v1 as workers

DASHBOARD_IP = os.environ['CDSW_IP_ADDRESS']

command = "ray start --head --block --include-dashboard=true --dashboard-port=$CDSW_READONLY_PORT --num-cpus=4 --num-gpus=1 &" 
subprocess.run(command, shell = True, executable="/bin/bash")

with open("RAY_HEAD_IP", 'w') as output_file:
    output_file.write(DASHBOARD_IP)
            
ray_head_addr = DASHBOARD_IP + ':6379'
ray_url = f"ray://{DASHBOARD_IP}:10001" 
worker_start_cmd = f"!ray start --block --address={ray_head_addr}"

time.sleep(7)
ray_workers = workers.launch_workers(
    n=1, 
    cpu=2, 
    memory=48,
    nvidia_gpu=1,
    code=worker_start_cmd,
)

os.system("vllm serve Llama-SEA-LION-v3.5-70B-R --port 8081 --tensor-parallel-size 2 > vllm.log 2>&1 &")