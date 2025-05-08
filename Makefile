# -----------------------------------------------------------------------------
#  NEBULA core – lightweight Makefile for edge devices (no Docker, no controller)
# -----------------------------------------------------------------------------

UV               := $(HOME)/.local/bin/uv
PYTHON_VERSION   := 3.11
UV_INSTALL_SCRIPT:= https://astral.sh/uv/install.sh
PATH             := $(HOME)/.local/bin:$(PATH)

# -----------------------------------------------------------------------------#
# 0. Helper: make sure “uv” is present                                         #
# -----------------------------------------------------------------------------#
.PHONY: check-uv
check-uv:                    ## Install uv if it is missing
	@if command -v $(UV) >/dev/null 2>&1; then \
		echo "📦 uv already installed"; \
	else \
		echo "📦 Installing uv…"; \
		curl -LsSf $(UV_INSTALL_SCRIPT) | sh; \
	fi; \
	if ! command -v $(UV) >/dev/null 2>&1; then \
		echo "❌ uv not found on PATH"; exit 1; \
	fi

# -----------------------------------------------------------------------------#
# 1. Python 3.11 managed by uv                                                 #
# -----------------------------------------------------------------------------#
.PHONY: install-python
install-python: check-uv      ## Download & pin Python $(PYTHON_VERSION)
	@echo "🐍 Installing Python $(PYTHON_VERSION)…"
	@$(UV) python install $(PYTHON_VERSION)
	@$(UV) python pin     $(PYTHON_VERSION)

# -----------------------------------------------------------------------------#
# 2. Project dependencies                                                      #
# -----------------------------------------------------------------------------#
.PHONY: install
install: install-python       ## Create .venv and install “core” deps
	@echo "📦 Syncing core dependencies…"
	@$(UV) sync --group core
	@echo "🔧 Installing pre-commit hooks (dev convenience)"
	@$(UV) run pre-commit install
	@echo "✅ Done."

# -----------------------------------------------------------------------------#
# 3. Enter the venv quickly                                                    #
# -----------------------------------------------------------------------------#
.PHONY: shell
shell:                         ## Spawn a shell with the venv activated
	@echo "🐚 Launching shell…"
	@if [ -n "$$VIRTUAL_ENV" ]; then \
		echo "Already inside venv: $$VIRTUAL_ENV"; \
	else \
		. .venv/bin/activate && bash --login; \
	fi

# -----------------------------------------------------------------------------#
# 4. Update repo & submodules                                                  #
# -----------------------------------------------------------------------------#
.PHONY: update
update:                        ## git pull + submodule update
	@git pull --ff-only
	@git submodule update --init --recursive

# -----------------------------------------------------------------------------#
# 5. Lockfile & quality checks (optional)                                      #
# -----------------------------------------------------------------------------#
.PHONY: lock
lock:                          ## Re-generate uv lock file
	@$(UV) lock

.PHONY: check
check:                         ## Run pre-commit on the whole repo
	@$(UV) run pre-commit run -a

.PHONY: check-plus
check-plus: check              ## black, mypy, deptry
	@$(UV) run black --check .
	@$(UV) run mypy
	@$(UV) run deptry .

# -----------------------------------------------------------------------------#
# 6. Build & clean                                                             #
# -----------------------------------------------------------------------------#
.PHONY: build
build: clean-build             ## Build the wheel
	@$(UV) build

.PHONY: clean-build
clean-build:                   ## Purge dist/ artifacts
	@rm -rf dist

.PHONY: clean
clean: clean-build             ## Purge caches & __pycache__
	@rm -rf __pycache__ */__pycache__ .mypy_cache

# -----------------------------------------------------------------------------#
# 7. Help                                                                      #
# -----------------------------------------------------------------------------#
.PHONY: help
help:                          ## Show this help
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	 | awk 'BEGIN {FS = ":.*?## "}; {printf "💡 \033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
