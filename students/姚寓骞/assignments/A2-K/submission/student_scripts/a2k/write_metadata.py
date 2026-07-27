from pathlib import Path

from student_scripts.a2k.common import metadata, require_cuda_and_limit_allocator, write_json

device, fraction = require_cuda_and_limit_allocator()
write_json(Path("local_results/a2k/run_metadata.json"), metadata(device, fraction, 0))
