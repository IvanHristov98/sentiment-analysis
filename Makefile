.PHONY: fmt
fmt:
	@black . -l 120

.PHONY: lint
lint:
	@echo "\033[0;33mLinting sentiment package...\033[0m"
	@pylint sentiment --max-line-length=120 --disable=missing-docstring --disable=too-many-locals --disable=super-init-not-called
