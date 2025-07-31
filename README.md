# Gradio LLM with vLLM/Ray Distributed GPU

<img width="809" height="294" alt="image" src="https://github.com/user-attachments/assets/2147ad88-f902-4f14-a6f0-9151d696897f" />

- The article describes how vLLM and Ray work together to perform high-throughput, distributed inference on the massive 70-billion-parameter `Llama-SEA-LION-v3.5-70B-R` model.
- vLLM uses Ray as its backend to manage the distributed worker processes, placing each tensor-parallel shard on the correct GPU across a multi-node cluster.
- This script uses tensor parallelism (tensor_parallel_size=2) to automatically shard the model's weights and computational graph across two GPUs, making it possible to run a model that far exceeds the VRAM of a single GPU. 🚀

🗒️ While you may load larger model size with more than 2 nodes, Tensor parallel workers can be spread out to more nodes which can degrade the performance unless you have fast interconnect across nodes, like Infiniband.

## Platform Requirement
☑️ Python 3.11/10

☑️ Cloudera AI(CAI)/Cloudera Machine Learning (CML) 1.5.x

## Procedure

1. Create a new CAI project.
   
2. Install python libraries.
  ```
  pip install vllm torch transformers ipywidgets gradio ray[default]
  ```

3. Download the pre-trained LLM into the project of the CAI/CML platform using either `git clone` or `wget`.
Example:
  ```
  git lfs clone https://huggingface.co/aisingapore/Llama-SEA-LION-v3.5-70B-R
  ```

4. Create Application to expose vLLM API endpoint. As VLLM utilizes Ray, this application will also host the Ray HEAD and Ray dashboard with 1 GPU device.
<img width="460" height="730" alt="image" src="https://github.com/user-attachments/assets/d128c611-969d-4e01-9fe4-54a73f9db055" />

5. Create another Application `gradio-app` with [app-gradio.py](app-gradio.py) to host the Gradio UI with a session profile without GPU. This Gradio UI will be communicating with the exposed API endpoint that runs vLLM and Ray in `vllm-api` Application. In total, there are 2 Applications.
<img width="1042" height="318" alt="image" src="https://github.com/user-attachments/assets/606998b7-ee9e-4552-a1bd-10c6065702ae" />

6. Start the `vllm-api` application and ensure that the model is fully loaded into the GPU before starting the `gradio-app` application. The code will spawn one Ray HEAD pod along with its associated worker pod within seconds.

```
NAME               READY   STATUS    RESTARTS   AGE     IP             NODE                                          NOMINATED NODE   READINESS GATES
16nhoob2jp0rc0dr   5/5     Running   0          7m36s   10.42.11.180   ares-ecs-ws-gpu03.ares.olympus.cloudera.com   <none>           <none>
dhw3fwp551sykyca   5/5     Running   0          7m13s   10.42.9.148    ares-ecs-ws-gpu01.ares.olympus.cloudera.com   <none>           <none>
```

7. Subsequently, the model will be loaded into the GPU VRAM. Depending on the size of the model, loading might take some time to complete. Once loaded, it is ready to serve queries. The following log shows model has been loaded successfully and vLLM application has started (vllm.log):
  ```
   Loading safetensors checkpoint shards:  83% Completed | 25/30 [1:23:11<15:58, 191.63s/it]
   Loading safetensors checkpoint shards:  87% Completed | 26/30 [1:26:52<13:21, 200.44s/it]
   Loading safetensors checkpoint shards:  90% Completed | 27/30 [1:28:01<08:02, 160.85s/it]
   Loading safetensors checkpoint shards:  93% Completed | 28/30 [1:32:03<06:10, 185.23s/it]
   Loading safetensors checkpoint shards:  97% Completed | 29/30 [1:35:56<03:19, 199.54s/it]
   Loading safetensors checkpoint shards: 100% Completed | 30/30 [1:40:29<00:00, 221.61s/it]
   ....  
  INFO:     Started server process [332]
  INFO:     Waiting for application startup.
  INFO:     Application startup complete.
  ```

8. Note that ray HEAD is initialized with GPU as required by vLLM. Therefore, ray HEAD node will be used as a worker with GPU.

  ```
  command = "ray start --head --block --include-dashboard=true --dashboard-port=$CDSW_READONLY_PORT --num-cpus=4 --num-gpus=1 &" 
  ```

9. When running [run-vllm.py](run-vllm.py) with single worker of 1 GPU, loading `Llama-SEA-LION-v3.5-70B-R` model will result in `torch.OutOfMemoryError: CUDA out of memory` error. This is because the model is too huge to fit in the VRAM of the GPU.
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

10. When running []() with 2 worker with 1 GPU each, loading `Llama-SEA-LION-v3.5-70B-R` model is successful. This is because the model is sharded into 2 GPU.
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

🗒️ `Llama-SEA-LION-v3.5-70B-R` supports a very long sequence length (context window) of 131,072 tokens. To handle this, vLLM needs to pre-allocate a large block of VRAM called the KV cache. To handle a 131,072 token sequence, it needs 20.00 GiB for the KV cache. Alternatively, you can restrict the maximum number of tokens the model can process in a single request. 
   
11. Click on the `vllm-api` link to view the Ray dashboard and verify that the model is fully loaded across 2 workers with GPU each.

<img width="900" height="621" alt="image" src="https://github.com/user-attachments/assets/5a21755a-2d73-4989-ae99-20999fc85fc5" />

12. Click on the `gradio-app` link to use the chat🤖.

<img width="900" height="747" alt="image" src="https://github.com/user-attachments/assets/187cdf45-576b-4307-b569-07be114355b2" />

Sample vllm.log:
```
INFO 07-31 09:46:04 [logger.py:41] Received request cmpl-7f77dd0c9abf4c6f839e81d88ef37188-0: prompt: 'Singapore is a', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=102, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: [128000, 92928, 374, 264], prompt_embeds shape: None, lora_request: None.
INFO 07-31 09:46:04 [async_llm.py:269] Added request cmpl-7f77dd0c9abf4c6f839e81d88ef37188-0.
INFO 07-31 09:46:11 [loggers.py:122] Engine 000: Avg prompt throughput: 0.4 tokens/s, Avg generation throughput: 2.6 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.2%, Prefix cache hit rate: 0.0%
INFO 07-31 09:46:21 [loggers.py:122] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 4.4 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.3%, Prefix cache hit rate: 0.0%
INFO:     10.42.7.21:41596 - "POST /v1/completions HTTP/1.1" 200 OK
```
