# Gradio LLM with vLLM/Ray

<img width="561" height="155" alt="image" src="https://github.com/user-attachments/assets/2f97f359-50fd-4260-99fc-6dcd367c30df" />

- The article describes how vLLM and Ray work together to perform high-throughput, distributed inference on the massive 70-billion-parameter `Llama-SEA-LION-v3.5-70B-R` model.
- vLLM uses Ray as its backend to manage the distributed worker processes, placing each tensor-parallel shard on the correct GPU across a multi-node cluster.
- This script uses tensor parallelism (tensor_parallel_size=2) to automatically shard the model's weights and computational graph across two GPUs, making it possible to run a model that far exceeds the VRAM of a single GPU. 🚀

🗒️ Tensor parallel workers can be spread out to 2+ nodes which can degrade the performance unless you have fast interconnect across nodes, like Infiniband.

## Platform Requirement
☑️ Python 3.11/10

☑️ Cloudera AI(CAI)/Cloudera Machine Learning (CML) 1.5.x

## Procedure

1. Create Application to expose vLLM API endpoint. As VLLM utilizes Ray, this application will also host the Ray dashboard with 1 GPU device.
<img width="460" height="730" alt="image" src="https://github.com/user-attachments/assets/d128c611-969d-4e01-9fe4-54a73f9db055" />

2. Create another Application to host the Gradio UI with session profile without GPU.
<img width="1042" height="318" alt="image" src="https://github.com/user-attachments/assets/606998b7-ee9e-4552-a1bd-10c6065702ae" />

3. Start `vllm-api` application and verify that the model is fully loaded into GPU before starting `gradio-app` application. Depending on the size of the model, loading might take some time to complete.
   
4. The following log shows model has been loaded successfully and vLLM application has started (vllm.log):
  ```
  INFO:     Started server process [332]
  INFO:     Waiting for application startup.
  INFO:     Application startup complete.
  ```

5. Note that ray HEAD is initialized with GPU as required by vLLM. Therefore, ray HEAD node will be used as a worker with GPU.

  ```
  command = "ray start --head --block --include-dashboard=true --dashboard-port=$CDSW_READONLY_PORT --num-cpus=4 --num-gpus=1 &" 
  ```

6. When running []() with single worker of 1 GPU, loading `Llama-SEA-LION-v3.5-70B-R` model will result in `torch.OutOfMemoryError: CUDA out of memory` error. This is because the model is too huge to fit in the VRAM of the GPU.
```
ray_workers = workers.launch_workers(
    n=0, 
    cpu=2, 
    memory=48,
    nvidia_gpu=1,
    code=worker_start_cmd,
)

os.system("vllm serve Llama-SEA-LION-v3.5-70B-R --port 8081 --tensor-parallel-size 1 > vllm.log 2>&1 &")
```

7. When running []() with 2 worker with 1 GPU each, loading `Llama-SEA-LION-v3.5-70B-R` model is successful. This is because the model is sharded into 2 GPU.
```
ray_workers = workers.launch_workers(
    n=1, 
    cpu=2, 
    memory=48,
    nvidia_gpu=1,
    code=worker_start_cmd,
)

os.system("vllm serve Llama-SEA-LION-v3.5-70B-R --port 8081 --tensor-parallel-size 2 > vllm.log 2>&1 &")
```


8. Click on the `vllm-api` link to view the Ray dashboard and verify that the model is fully loaded across 2 workers with GPU each.


<img width="1042" height="639" alt="image" src="https://github.com/user-attachments/assets/1fb544ce-591e-4f0c-9c3e-f4ee551b1f62" />

9. Click on the `gradio-app` link to use the chat🤖.

<img width="1042" height="746" alt="image" src="https://github.com/user-attachments/assets/bc3c0b97-f3e5-48eb-9ddd-fee05906e5ce" />

