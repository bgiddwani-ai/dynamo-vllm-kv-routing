# Dynamo + vLLM with KV routing

Run **Nemotron Super 49B FP8** on **two TP=4 vLLM workers (4+4 H200 GPUs)** behind Dynamo's KV-aware router. Uses pinned Dynamo **1.5.1**, vLLM **0.28.0**, FP8 KV cache, prefix caching and KV events.

## Requirements

- Kubernetes **1.34+**, Helm 3, Python 3.12, NVIDIA drivers/container runtime and the [NVIDIA GPU DRA driver](https://github.com/NVIDIA/k8s-dra-driver-gpu#installation).
- Eight available H200 GPUs; at least 36 CPU cores and 97 GiB RAM for workload requests, plus platform overhead.
- A model PVC shared by the pods, at least 60 GB free for weights, registry access and permission to allocate eight GPUs. Platform installation requires cluster-admin access.

## 1. Connect and clone

```bash
ssh -t USER@GPU_NODE
screen -S dynamo-kv
git clone https://github.com/bgiddwani-ai/dynamo-vllm-kv-routing.git
cd dynamo-vllm-kv-routing
export NAMESPACE=llm-demo
export DYNAMO_NAMESPACE=dynamo-system
export RELEASE=nemotron
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
```

Use your own namespace and cluster context. Detach screen with **Ctrl-A, D**; reconnect with `screen -r dynamo-kv`.

## 2. Install official Dynamo

Skip this step if a compatible operator is already installed. Otherwise, install from [NVIDIA's official repository](https://github.com/ai-dynamo/dynamo):

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

This installs the operator, CRDs, Grove and KAI Scheduler. See the [official installation guide](https://github.com/ai-dynamo/dynamo/blob/v1.5.1/docs/fern/pages/kubernetes/installation/install-dynamo.md) for existing Grove/KAI installations.

## 3. Download the model

Set `MODEL_DIR` on the filesystem exposed by your PVC. With PVC root `/shared`, the example below matches the chart's default `model.subPath`.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export MODEL_DIR=/shared/models/Llama-3_3-Nemotron-Super-49B-v1_5-FP8
# If required: .venv/bin/hf auth login
bash scripts/download_model.sh
```

The script downloads the pinned model revision and validates its safetensors shards.

## 4. Configure GPUs and storage

```bash
kubectl get nodes -o wide
kubectl -n "$NAMESPACE" get resourcequota,pvc
nvidia-smi --query-gpu=index,uuid,memory.used --format=csv
id -u
id -g
cp values.yaml values.local.yaml
${EDITOR:-vi} values.local.yaml
```

Set `node` to the Kubernetes hostname; set `model.pvc`, `model.subPath` and `user.uid/gid` for your storage. Replace the first worker group's UUIDs with GPUs **0–3**, and the second with GPUs **4–7**. Each group needs four distinct, available GPUs and `replicas: 1`.

## 5. Deploy

```bash
helm lint . -f values.local.yaml
helm template "$RELEASE" . -n "$NAMESPACE" -f values.local.yaml > rendered.yaml
kubectl -n "$NAMESPACE" apply --dry-run=server -f rendered.yaml
helm upgrade --install "$RELEASE" . -n "$NAMESPACE" \
  --reset-values -f values.local.yaml
kubectl -n "$NAMESPACE" get pods -o wide --watch
```

Wait for the frontend and both workers to become Ready; press Ctrl-C to leave the watch. The chart runs `dynamo.frontend --router-mode kv` and `dynamo.vllm` with prefix caching and KV-event publication enabled. Requests must go through the Dynamo frontend.

## 6. Test

```bash
kubectl -n "$NAMESPACE" port-forward "service/$RELEASE-kv" 8000:8000 \
  > /tmp/dynamo-kv-port-forward.log 2>&1 &
curl --fail --retry 20 --retry-connrefused --retry-delay 1 \
  http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Llama-3_3-Nemotron-Super-49B-v1_5-FP8","messages":[{"role":"user","content":"Explain KV caching in two sentences."}],"max_tokens":128,"stream":true}'
```

KV routing helps when requests reuse long, token-identical prefixes across multiple replicas. Put the changing question after the shared document.

## Performance: 4+4 H200 GPUs

**RR:** native vLLM + NGINX round-robin. **KV:** Dynamo + vLLM with KV routing. Both used the same model, runtime, GPUs and engine settings, with prefix caching enabled. RR ran first, followed by fresh KV workers; caches reset before each concurrency cell.

| Input profile | Median p50 TTFT change | Median output throughput change |
|---|---:|---:|
| 1,024 tokens | +2.06% | −0.77% |
| 8,192 tokens | +3.35% | −0.11% |
| Short/medium combined | +2.83% | −0.28% |
| 24,128 tokens, repeated prefixes | −30.60% | +16.42% |

Changes are medians of paired case percentages: **lower TTFT** and **higher throughput** are better. Throughput is aggregate generated output tokens/s across both workers.

**Short/medium:** 1K/8K inputs, 128 output tokens, 0%/90% prefix sharing, two trials per case. **Long:** 256 documents × three questions, 24,000-token prefix + 128-token question, 64 output tokens, one trial per case. Both tested concurrency **1, 8, 32, 64, 128, 200 and 256**.

Long-context absolute results:

| Concurrency | RR p50 TTFT (ms) | KV p50 TTFT (ms) | RR output (tok/s) | KV output (tok/s) |
|---:|---:|---:|---:|---:|
| 1 | 1,065.7 | 80.0 | 59.9 | 79.8 |
| 8 | 1,092.9 | 1,057.6 | 168.8 | 177.2 |
| 32 | 1,917.7 | 1,413.9 | 192.4 | 214.9 |
| 64 | 2,313.0 | 2,092.4 | 195.6 | 217.1 |
| 128 | 5,392.4 | 3,446.6 | 202.2 | 236.6 |
| 200 | 16,731.8 | 11,612.7 | 208.6 | 242.9 |
| 256 | 33,359.8 | 21,089.3 | 209.2 | 249.0 |

Full measurements: [28 short/medium cases](benchmarks/short-context.csv) · [7 long-context cases](benchmarks/long-context.csv).

Long-context p95 TTFT was essentially unchanged. The concurrency-1 median includes warm-prefix hits; cold-prefill latency was similar. These finite-batch results compare complete serving stacks and do not isolate the router's contribution.
