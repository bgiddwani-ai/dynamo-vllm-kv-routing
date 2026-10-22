"""Validate the pinned download and every safetensors shard without loading weights."""
import collections
import json
import pathlib
import struct
import sys

root = pathlib.Path(sys.argv[1])
index = json.loads((root / 'model.safetensors.index.json').read_text())
required = set(index['weight_map'].values())
seen = set()
dtypes = collections.Counter()
total_bytes = 0
for name in sorted(required):
    path = root / name
    with path.open('rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        if not 0 < header_len < 100_000_000:
            raise ValueError(f'Invalid safetensors header: {name}')
        header = json.loads(f.read(header_len))
    data_size = path.stat().st_size - 8 - header_len
    ends = []
    for tensor, info in header.items():
        if tensor == '__metadata__':
            continue
        begin, end = info['data_offsets']
        assert 0 <= begin <= end <= data_size, (name, tensor)
        assert index['weight_map'][tensor] == name, tensor
        assert tensor not in seen, tensor
        seen.add(tensor)
        ends.append(end)
        dtypes[info['dtype']] += 1
        total_bytes += end - begin
    assert max(ends) == data_size, name
assert seen == set(index['weight_map'])
if 'total_size' in index.get('metadata', {}):
    assert total_bytes == index['metadata']['total_size']
print(json.dumps({'status': 'validated', 'shards': len(required), 'tensors': len(seen),
                  'tensor_bytes': total_bytes, 'dtypes': dict(dtypes)}, indent=2))
