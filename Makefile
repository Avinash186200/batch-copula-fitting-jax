.PHONY: install test bench speed speed-multicore report

install:
	pip install -c constraints.txt -e ".[dev,jax]"

test:
	pytest -q

# Offline, deterministic, held-out seeds: recovery, agreement, selection, tail, demo.
# Writes results/*.json and RESULTS.md. CI reruns this and compares with the committed JSON.
bench:
	python -m copulabatch.cli bench --out results

# Machine-specific throughput; writes results/speed_<label>.json and re-renders RESULTS.md.
speed:
	python -m copulabatch.cli speed --out results --label local

# The same, with JAX spreading pairs over 16 CPU "devices" (one per core) via pmap.
speed-multicore:
	python -m copulabatch.cli speed --out results --label local-16dev --engines jax --cpu-devices 16

report:
	python -m copulabatch.cli report --out results
