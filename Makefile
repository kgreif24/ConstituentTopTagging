.PHONY: venv
.PHONY: update
.PHONY: test

# Changing shell to bash
SHELL := /bin/bash

venv:
	source bin/setup.sh && make test
	@echo "[INFO] [BUILD] Done. Running test"
	$(MAKE) -C lib/preprocessing
	$(MAKE) -C lib/shuffle

test:
	install/test.py
