# Config
ENV_NAME := my_project_env
# Adjust this specific path if your envs are stored elsewhere (e.g. ~/anaconda3/envs)
ENV_PYTHON := /usr/local/anaconda/envs/$(ENV_NAME)/bin/python
ENV_PIP    := /usr/local/anaconda/envs/$(ENV_NAME)/bin/pip

.PHONY: all create-env install simulate test clean profile complexity benchmark parallel stability-check

all: simulate

# --- Environment ---
create-env:
	conda create -y -n $(ENV_NAME) python=3.11

install:
	$(ENV_PIP) install -q -U pip
	@test -f requirements.txt && $(ENV_PIP) install -q -r requirements.txt || true

# --- Main Tasks ---
simulate: install
	$(ENV_PYTHON) -m src.simulation

test: install
	$(ENV_PYTHON) -m tests.function_test
	$(ENV_PYTHON) -m tests.data_test
	$(ENV_PYTHON) -m tests.reproducibility_test

# --- Analysis Tasks ---
profile: install
	$(ENV_PYTHON) -m src.simulation --profile_mode

complexity: install
	$(ENV_PYTHON) scripts/plot_complexity.py

benchmark: install
	$(ENV_PYTHON) scripts/plot_benchmark.py

parallel: install
	$(ENV_PYTHON) src/simulation.py --parallel --n_cores 4

stability-check: install
	$(ENV_PYTHON) scripts/check_stability.py

# --- Cleanup ---
clean:
	rm -f results/raw/* results/figures/* *.prof *.log
