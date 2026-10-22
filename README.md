# Dynamo + vLLM with KV routing

Deploy **Nemotron Super 49B FP8** using Helm: one Dynamo KV router → **two vLLM workers, TP=4 each**, on eight H200 GPUs. The router uses cached-prefix information and worker load to place requests. Both workers enable prefix caching and publish KV events.

This recipe packages the tested **Dynamo runtime 1.5.1 / vLLM 0.28.0** configuration. Install the platform from the [official NVIDIA Dynamo repository](https://github.com/ai-dynamo/dynamo) below. The source release, runtime image and model revision are pinned; revalidate upgrades separately.

## Measured performance: 4+4 GPUs

Measured on **22 October 2026**, using **eight H200 GPUs as two TP=4 replicas** with the same Nemotron FP8 model, pinned runtime, engine settings and physical GPU groups in both passes. **RR** means native vLLM behind NGINX round-robin; **KV** means Dynamo vLLM behind its KV-aware frontend. The native pass finished first, then its workers were removed before fresh KV workers launched. Both stacks enabled prefix caching; caches reset before each concurrency cell.

### Median changes across paired cases

TTFT is time to first token. Below, a **negative TTFT change is better** and a **positive throughput change is better**. Changes are `(KV − RR) / RR × 100`; they are medians of paired case changes, not changes between pooled request medians.

| Input profile | Median p50 TTFT change | Median output throughput change |
|---|---:|---:|
| 1,024 tokens — short | +2.06% | -0.77% |
| 8,192 tokens — medium | +3.35% | -0.11% |
| Short/medium combined — 28 cases | +2.83% | -0.28% |
| 24,128 tokens — long | -30.60% | +16.42% |

The short/medium matrix showed **no overall improvement**. The long-context workload showed **30.60% lower p50 TTFT** and **16.42% higher output throughput**; its median paired p95 TTFT reduction was only **0.065%**, essentially unchanged.

### Short and medium context: 1K / 8K input

**23,808 measured requests, 112 cells, zero errors.** Inputs were exactly 1,024 or 8,192 tokens; each response generated 128 tokens. Each input length tested 0% and 90% configured prefix sharing across eight prefix families, at concurrency **1, 8, 32, 64, 128, 200 and 256**. Each case ran two trials per backend, with `max(64, 2 × concurrency)` requests per trial. The absolute values below average the two per-trial statistics; request quantiles are not pooled.

<details>
<summary>Expand all 28 short/medium cases: absolute TTFT and output throughput</summary>

| Input tokens | Shared prefix | Concurrency | RR p50 TTFT (ms) | KV p50 TTFT (ms) | RR output (tok/s) | KV output (tok/s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 0% | 1 | 49.4 | 51.5 | 157.4 | 156.4 |
| 1,024 | 0% | 8 | 158.6 | 158.1 | 1,096.4 | 1,093.5 |
| 1,024 | 0% | 32 | 575.2 | 573.6 | 2,900.2 | 2,875.8 |
| 1,024 | 0% | 64 | 745.4 | 669.9 | 3,878.1 | 3,854.5 |
| 1,024 | 0% | 128 | 1,490.0 | 1,212.8 | 5,071.4 | 4,903.8 |
| 1,024 | 0% | 200 | 1,949.2 | 2,156.4 | 5,484.5 | 5,446.0 |
| 1,024 | 0% | 256 | 2,104.5 | 2,324.7 | 5,748.7 | 5,531.0 |
| 1,024 | 90% | 1 | 20.1 | 45.4 | 161.9 | 161.5 |
| 1,024 | 90% | 8 | 36.1 | 51.7 | 1,189.2 | 1,185.7 |
| 1,024 | 90% | 32 | 129.1 | 126.3 | 3,840.5 | 3,608.5 |
| 1,024 | 90% | 64 | 205.7 | 215.3 | 6,304.6 | 6,281.7 |
| 1,024 | 90% | 128 | 364.0 | 324.8 | 11,218.3 | 11,078.9 |
| 1,024 | 90% | 200 | 473.1 | 504.5 | 13,843.7 | 13,676.3 |
| 1,024 | 90% | 256 | 634.2 | 608.3 | 16,044.3 | 14,789.2 |
| 8,192 | 0% | 1 | 312.6 | 307.4 | 116.7 | 116.5 |
| 8,192 | 0% | 8 | 1,039.5 | 1,030.9 | 511.9 | 510.8 |
| 8,192 | 0% | 32 | 2,661.2 | 2,501.5 | 724.0 | 722.9 |
| 8,192 | 0% | 64 | 3,077.9 | 2,942.4 | 771.6 | 771.9 |
| 8,192 | 0% | 128 | 2,569.7 | 3,992.1 | 806.6 | 807.9 |
| 8,192 | 0% | 200 | 2,391.7 | 3,876.5 | 812.4 | 811.9 |
| 8,192 | 0% | 256 | 2,484.5 | 6,626.9 | 819.2 | 817.6 |
| 8,192 | 90% | 1 | 78.2 | 68.3 | 139.4 | 144.1 |
| 8,192 | 90% | 8 | 95.9 | 87.5 | 796.6 | 835.1 |
| 8,192 | 90% | 32 | 477.0 | 607.8 | 1,576.9 | 1,680.6 |
| 8,192 | 90% | 64 | 877.9 | 1,055.3 | 2,441.7 | 2,506.8 |
| 8,192 | 90% | 128 | 1,856.4 | 2,037.9 | 3,560.0 | 3,541.6 |
| 8,192 | 90% | 200 | 2,517.9 | 2,549.6 | 3,977.9 | 3,969.0 |
| 8,192 | 90% | 256 | 2,420.9 | 2,552.4 | 4,427.1 | 4,266.6 |

</details>

[Download all short/medium measurements](benchmarks/short-context.csv), including mean/p95 TTFT, end-to-end latency and throughput.

### Long context: 24K reusable documents

**10,752 measured requests, 14 cells, zero errors.** The workload used 256 PG-19 book excerpts, each with a **24,000-token document prefix + 128-token question**, followed by 64 generated tokens. Three different question rounds gave 768 requests per cell and one exploratory trial per backend/concurrency. The 6,144,000-token prefix set exceeded one worker's measured cache capacity but fit within the two workers' combined capacity. This creates an opportunity for routing to avoid duplicated prefixes; cache memory remains local to each worker.

Throughput below is aggregate **generated output tokens per second across both workers**, including the cell's cold and repeated-prefix requests; it is not input-token throughput or a per-GPU figure.

| Concurrency | RR p50 TTFT (ms) | KV p50 TTFT (ms) | TTFT change | RR output (tok/s) | KV output (tok/s) | Throughput change |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,065.7 | 80.0 | -92.49% | 59.9 | 79.8 | +33.12% |
| 8 | 1,092.9 | 1,057.6 | -3.24% | 168.8 | 177.2 | +5.00% |
| 32 | 1,917.7 | 1,413.9 | -26.27% | 192.4 | 214.9 | +11.74% |
| 64 | 2,313.0 | 2,092.4 | -9.54% | 195.6 | 217.1 | +10.95% |
| 128 | 5,392.4 | 3,446.6 | -36.08% | 202.2 | 236.6 | +17.01% |
| 200 | 16,731.8 | 11,612.7 | -30.60% | 208.6 | 242.9 | +16.42% |
| 256 | 33,359.8 | 21,089.3 | -36.78% | 209.2 | 249.0 | +19.06% |

[Download all long-context measurements](benchmarks/long-context.csv), including mean/p95 TTFT, end-to-end latency, throughput and cache-hit counters. The same aggregate output-throughput definition applies to the short/medium table.

**Interpretation and limits:** At concurrency 1, long-context prefix-token hits rose from 40.15% to 66.31%; cold-round p50 TTFT remained similar (RR 1,071 ms; KV 1,054 ms). The large overall p50 reduction reflects more requests hitting warm prefixes, not a 92% speedup of cold prefill. These are finite-batch, fixed native-first, complete-serving-stack comparisons; the results do not isolate the router's contribution. Short/medium used two trials and long used one: no statistical-significance or answer-quality claim is made. Output lengths differ between profiles, so compare RR versus KV within each profile.

## Prerequisites

- Kubernetes **1.34+**, Helm 3 and Python 3.12, with NVIDIA drivers, container runtime and NVIDIA GPU **DRA** driver configured. Tested on Kubernetes 1.34.3 with Helm 3.18.3 and Python 3.12.3. See [NVIDIA's DRA installation guide](https://github.com/NVIDIA/k8s-dra-driver-gpu#installation); installing Dynamo does not install the GPU DRA driver.
- Cluster-admin access for the one-time Dynamo platform installation, or an administrator who provides a compatible existing installation.
- One healthy eight-GPU H200 node; permission/quota to allocate eight GPUs in your namespace. A device-plugin-only cluster needs a different GPU allocation template.
- An existing model PVC mountable by all three pods, at least **60 GB** free for weights, and permission to pull the NVIDIA runtime image. Configure a namespace image-pull secret/service account if your registry requires one.
- CPU/RAM headroom for the chart requests: 36 CPU cores and 97 GiB RAM total, plus cluster/operator overhead. Each worker has a 96 GiB memory limit and 16 GiB shared memory.

## 1. Open a persistent shell

```bash
ssh -t USER@GPU_NODE
screen -S dynamo-kv
git clone https://github.com/bgiddwani-ai/dynamo-vllm-kv-routing.git
cd dynamo-vllm-kv-routing
export NAMESPACE=llm-demo
export DYNAMO_NAMESPACE=dynamo-system
export RELEASE=nemotron
kubectl config current-context
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
```

Run the remaining commands inside this screen. Detach with **Ctrl-A, D**; reconnect with `screen -r dynamo-kv`.

## 2. Install Dynamo from the official repository

For a fresh cluster, install the operator and platform from NVIDIA's pinned source release. On a shared cluster, first check `helm list --all-namespaces` with your administrator; **reuse an existing compatible operator instead of installing a second one**. If it is already installed, skip to step 3.

```bash
export DYNAMO_VERSION=1.5.1
export DYNAMO_SOURCE_DIR="$HOME/dynamo-official-$DYNAMO_VERSION"
git clone --depth 1 --branch "v$DYNAMO_VERSION" \
  https://github.com/ai-dynamo/dynamo.git "$DYNAMO_SOURCE_DIR"
helm dependency build "$DYNAMO_SOURCE_DIR/deploy/helm/charts/platform"
helm install dynamo-platform "$DYNAMO_SOURCE_DIR/deploy/helm/charts/platform" \
  --namespace "$DYNAMO_NAMESPACE" --create-namespace \
  --set global.grove.install=true \
  --set global.kai-scheduler.install=true \
  --set-string 'kai-scheduler.scheduler.args.default-staleness-grace-period=-1s' \
  --wait --timeout 10m
kubectl -n "$DYNAMO_NAMESPACE" get pods
kubectl get crd dynamographdeployments.nvidia.com
kubectl get deviceclass gpu.nvidia.com
```

The platform installs the Dynamo operator, which applies the CRDs automatically. These flags also install the compatible bundled Grove and KAI Scheduler for this example. Production clusters can manage Grove/KAI separately and use `global.grove.enabled=true` / `global.kai-scheduler.enabled=true` instead of the corresponding `install` flags. This release uses Kubernetes discovery and TCP/ZMQ by default, so this setup does not require bundled etcd or NATS. See the [official installation guide at v1.5.1](https://github.com/ai-dynamo/dynamo/blob/v1.5.1/docs/fern/pages/kubernetes/installation/install-dynamo.md).

## 3. Download the pinned model

`MODEL_DIR` must be on the shared filesystem exposed by your PVC. For example, if the PVC root is `/shared`, use the path below and keep `model.subPath: models/Llama-3_3-Nemotron-Super-49B-v1_5-FP8`.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export MODEL_DIR=/shared/models/Llama-3_3-Nemotron-Super-49B-v1_5-FP8
# If authentication is requested: .venv/bin/hf auth login
bash scripts/download_model.sh
```

The download pins revision `04822723e77e036ddf2d24e83c6d469d3b009252`. Validation checks the safetensors index, shard headers and tensor bounds without loading weights into RAM. An existing download can be checked with `.venv/bin/python scripts/verify_model.py "$MODEL_DIR"`. The chart uses `--trust-remote-code` for this pinned NVIDIA model; keep the downloaded directory unchanged.

## 4. Set node, storage and GPU ownership

```bash
kubectl get nodes -o wide
kubectl get crd dynamographdeployments.nvidia.com
kubectl get deviceclass gpu.nvidia.com
kubectl -n "$NAMESPACE" get resourcequota,pvc
nvidia-smi --query-gpu=index,uuid,memory.used,temperature.gpu --format=csv
id -u
id -g
cp values.yaml values.local.yaml
${EDITOR:-vi} values.local.yaml
```

Edit these values:

| Value | Set to |
|---|---|
| `node` | Kubernetes hostname of the GPU node, not its SSH address |
| `model.pvc` | Existing PVC in `$NAMESPACE` |
| `model.subPath` | Model directory relative to the PVC root |
| `user.uid`, `user.gid` | Your `id -u` / `id -g`, or other IDs allowed to read that directory |
| `workerGroups[0].gpuUUIDs` | Four real GPU UUIDs corresponding to GPUs 0–3 |
| `workerGroups[1].gpuUUIDs` | Four different UUIDs corresponding to GPUs 4–7 |

Use available GPUs only. Each group has **one** replica; add another group with disjoint GPUs to scale. DRA enforces device ownership; a startup guard verifies the assigned UUIDs and refuses busy devices. Do not override `NVIDIA_VISIBLE_DEVICES`. The sample UUIDs are placeholders and must be replaced.

## 5. Validate and deploy with Helm

```bash
helm lint . -f values.local.yaml
helm template "$RELEASE" . -n "$NAMESPACE" -f values.local.yaml > rendered.yaml
kubectl -n "$NAMESPACE" apply --dry-run=server -f rendered.yaml
helm upgrade --install "$RELEASE" . -n "$NAMESPACE" \
  --reset-values -f values.local.yaml
kubectl -n "$NAMESPACE" get pods -o wide --watch
```

Wait for the **frontend and both workers** to become Ready, then press Ctrl-C to leave the watch. The operator may create additional infrastructure pods. A successful Helm command alone does not mean model loading is complete. Inspect a failing pod using `kubectl -n "$NAMESPACE" describe pod POD` and `kubectl -n "$NAMESPACE" logs POD -c main`.

All namespace and release names above are examples. Substitute your own Kubernetes context, namespace, node name, PVC and GPU UUIDs; no private cluster addresses or credentials are required in this repository. The chart sets `runtimeVersionOverride: "1.5.1"` so the operator can validate the digest-pinned runtime image.

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

## 6. Send a request

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

References: [Official platform chart](https://github.com/ai-dynamo/dynamo/tree/v1.5.1/deploy/helm/charts/platform), [Dynamo source](https://github.com/ai-dynamo/dynamo), [model card](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8).
