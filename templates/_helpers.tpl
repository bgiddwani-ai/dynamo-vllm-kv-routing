{{/* Prevent overlapping fixed GPU assignments and accidental replica duplication. */}}
{{- define "dynamo.validate" -}}
{{- if lt (int .Values.workers.tensorParallel) 1 -}}
  {{- fail "workers.tensorParallel must be positive" -}}
{{- end -}}
{{- if not .Values.workerGroups -}}
  {{- fail "At least one worker group is required" -}}
{{- end -}}
{{- $seen := dict -}}
{{- $names := dict -}}
{{- $claims := dict -}}
{{- range .Values.workerGroups -}}
  {{- if ne (int .replicas) 1 -}}
    {{- fail "Each fixed-GPU group must have replicas: 1; add a disjoint group to scale" -}}
  {{- end -}}
  {{- if ne (len .gpuUUIDs) (int $.Values.workers.tensorParallel) -}}
    {{- fail "Each gpuUUIDs list must have exactly tensorParallel entries" -}}
  {{- end -}}
  {{- if or (hasKey $names .name) (hasKey $claims .claimSuffix) -}}
    {{- fail "Worker group names and claimSuffix values must be unique" -}}
  {{- end -}}
  {{- $_ := set $names .name true -}}
  {{- $_ := set $claims .claimSuffix true -}}
  {{- range .gpuUUIDs -}}
    {{- if not (regexMatch "^GPU-[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$" .) -}}
      {{- fail "Use full GPU UUIDs from nvidia-smi" -}}
    {{- end -}}
    {{- if hasKey $seen . -}}
      {{- fail "GPU UUIDs must be unique across all worker groups" -}}
    {{- end -}}
    {{- $_ := set $seen . true -}}
  {{- end -}}
{{- end -}}
{{- end -}}
