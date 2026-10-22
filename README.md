# Dynamo + vLLM with KV routing

Deploy **Nemotron Super 49B FP8** using Helm: one Dynamo KV router → **two vLLM workers, TP=4 each**, on eight H200 GPUs. The router uses cached-prefix information and worker load to place requests. Both workers enable prefix caching and publish KV events.

This recipe packages the tested **Dynamo runtime 1.5.1 / vLLM 0.28.0** configuration. The runtime image and model revision are pinned; the installed operator must be compatible. Revalidate upgrades separately.

## Prerequisites

- Kubernetes with the Dynamo operator/Grove and NVIDIA GPU **DRA** driver already installed. Tested on Kubernetes 1.34.3 with Helm 3.18.3 and Python 3.12.3.
- One healthy eight-GPU H200 node; permission/quota to allocate eight GPUs in your namespace. A device-plugin-only cluster needs a different GPU allocation template.
- An existing model PVC mountable by all three pods, at least **60 GB** free for weights, and permission to pull the NVIDIA runtime image. Configure a namespace image-pull secret/service account if your registry requires one.
- CPU/RAM headroom for the chart requests: 36 CPU cores and 97 GiB RAM total, plus cluster/operator overhead. Each worker has a 96 GiB memory limit and 16 GiB shared memory.

## 1. Open a persistent shell

```bash
ssh -t USER@GPU_NODE
screen -S dynamo-kv
git clone https://github.com/bgiddwani-ai/dynamo-vllm-kv-routing.git
cd dynamo-vllm-kv-routing
export NAMESPACE=YOUR_NAMESPACE
export RELEASE=nemotron
```

Run the remaining commands inside this screen. Detach with **Ctrl-A, D**; reconnect with `screen -r dynamo-kv`.

## 2. Download the pinned model

`MODEL_DIR` must be on the shared filesystem exposed by your PVC. For example, if the PVC root is `/shared`, use the path below and keep `model.subPath: models/Llama-3_3-Nemotron-Super-49B-v1_5-FP8`.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export MODEL_DIR=/shared/models/Llama-3_3-Nemotron-Super-49B-v1_5-FP8
# If authentication is requested: .venv/bin/hf auth login
bash scripts/download_model.sh
```

The download pins revision `04822723e77e036ddf2d24e83c6d469d3b009252`. Validation checks the safetensors index, shard headers and tensor bounds without loading weights into RAM. An existing download can be checked with `.venv/bin/python scripts/verify_model.py "$MODEL_DIR"`. The chart uses `--trust-remote-code` for this pinned NVIDIA model; keep the downloaded directory unchanged.

## 3. Set node, storage and GPU ownership

```bash
kubectl get nodes -o wide
kubectl get crd dynamographdeployments.nvidia.com
kubectl get deviceclass gpu.nvidia.com
kubectl -n "$NAMESPACE" get resourcequota,pvc
nvidia-smi --query-gpu=index,uuid,memory.used,temperature.gpu --format=csv
cp values.yaml values.local.yaml
${EDITOR:-vi} values.local.yaml
```

Edit these values:

| Value | Set to |
|---|---|
| `node` | Kubernetes hostname of the GPU node, not its SSH address |
| `model.pvc` | Existing PVC in `$NAMESPACE` |
| `model.subPath` | Model directory relative to the PVC root |
| `user.uid`, `user.gid` | IDs allowed to read that directory |
| `workerGroups[0].gpuUUIDs` | Four real GPU UUIDs corresponding to GPUs 0–3 |
| `workerGroups[1].gpuUUIDs` | Four different UUIDs corresponding to GPUs 4–7 |

Use available GPUs only. Each group has **one** replica; add another group with disjoint GPUs to scale. DRA enforces device ownership; a startup guard verifies the assigned UUIDs and refuses busy devices. Do not override `NVIDIA_VISIBLE_DEVICES`. The sample UUIDs are placeholders and must be replaced.

## 4. Validate and deploy with Helm

```bash
helm lint . -f values.local.yaml
helm template "$RELEASE" . -n "$NAMESPACE" -f values.local.yaml > rendered.yaml
kubectl -n "$NAMESPACE" apply --dry-run=server -f rendered.yaml
helm upgrade --install "$RELEASE" . -n "$NAMESPACE" \
  --reset-values -f values.local.yaml
kubectl -n "$NAMESPACE" get pods \
  -l "nic.nvidia.com/release=$RELEASE" -o wide --watch
```

Wait for the **frontend and both workers** to become Ready, then press Ctrl-C to leave the watch. The operator may create additional infrastructure pods. A successful Helm command alone does not mean model loading is complete. Inspect a failing pod using `kubectl -n "$NAMESPACE" describe pod POD` and `kubectl -n "$NAMESPACE" logs POD -c main`.

The chart launches these entry points:

```text
Frontend: python3 -m dynamo.frontend --router-mode kv --http-port 8000
Workers:  python3 -m dynamo.vllm --model /model-store ...
```

The worker settings are TP=4, 32,000-token context, ModelOpt FP8 weights, FP8 KV cache, 0.88 GPU memory utilization, chunked prefill, 16,384 batched tokens, 128 sequences per worker, and 64-token cache blocks. The two switches that connect prefix caching to the router are:

```text
--enable-prefix-caching
--kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}'
```

Use the Dynamo frontend endpoint for requests. Launching a standalone vLLM API server does not enable Dynamo KV routing. Keep operator-managed discovery and KV-event traffic reachable between pods.

## 5. Send a request

```bash
kubectl -n "$NAMESPACE" port-forward "service/$RELEASE-kv" 8000:8000 \
  > /tmp/dynamo-kv-port-forward.log 2>&1 &
curl --fail --retry 20 --retry-connrefused --retry-delay 1 \
  http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Llama-3_3-Nemotron-Super-49B-v1_5-FP8","messages":[{"role":"user","content":"Explain KV caching in two sentences."}],"max_tokens":128,"stream":true}'
```

For a useful routing test, reuse a long, **token-identical document prefix** across different questions and multiple replicas. Change the question after the document. Compare the same prompts, request order, generation length and concurrency, with fresh deployments and equal cache warmup. Long inputs with no reusable prefix may show little benefit. At high load, balancing queues can outweigh cache affinity; check p95 latency as well as median TTFT and throughput.

## Operations

The worker guard starts below **60°C** and stops its own worker at **80°C**, thermal throttling or failed GPU telemetry. A trip creates a latch in that pod's `/cache`; a container restart will not resume inference. Inspect logs and fix the cause before deliberately recreating the pod. These checks reduce risk and cannot guarantee host stability. Keep the screen and deployment running until you are finished.

To release GPUs later, explicitly run:

```bash
helm uninstall "$RELEASE" -n "$NAMESPACE"
kubectl -n "$NAMESPACE" get pods,resourceclaims
```

Confirm the worker pods and claims disappear and GPU memory is released. The model PVC and weights are retained. Do not shut down a shared node.

References: [Dynamo operator](https://docs.nvidia.com/dynamo/latest/kubernetes-deployment/deployment-guide/dynamo-operator), [Dynamo source](https://github.com/ai-dynamo/dynamo), [model card](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8).
