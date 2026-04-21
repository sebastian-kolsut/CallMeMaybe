VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
FLAKE8 = $(VENV)/bin/flake8
MYPY = $(VENV)/bin/mypy

.PHONY: install run debug clean lint

$(VENV)/bin/activate:
	python3 -m venv $(VENV)

install: $(VENV)/bin/activate
#	$(PIP) install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
	$(PIP) install -e . flake8 mypy

run:
	$(PYTHON) -m src

debug:
	$(PYTHON) -m pdb src/__main__.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .mypy_cache .pytest_cache build dist *.egg-info $(VENV)

lint:
	$(FLAKE8) src
	$(MYPY) src --follow-imports=skip --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
