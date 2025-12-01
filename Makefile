VENV   := .venv
PY     := $(VENV)/bin/python3
PIP    := $(VENV)/bin/pip3

.PHONY: all simulate venv install clean test

all: simulate

venv:
	@test -d $(VENV) || python3 -m venv $(VENV)

install: venv
	@$(PIP) -q install -U pip
	@test -f requirements.txt && $(PIP) -q install -r requirements.txt || true

simulate: install
	@$(PY) -m src.simulation

test: install
	@$(PY) -m tests.function_test
	@$(PY) -m tests.data_test
	@$(PY) -m tests.reproducibility_test

clean:
	@rm results/raw/* results/figures/*
