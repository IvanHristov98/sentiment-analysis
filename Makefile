.PHONY: fmt
fmt:
	@pipenv run black . -l 120

.PHONY: lint
lint:
	@echo "\033[0;33mLinting sentiment package...\033[0m"
	@pipenv run pylint sentiment --max-line-length=120 --disable=missing-docstring --disable=too-many-locals --disable=super-init-not-called

.PHONY: install
install:
	@pipenv install
	@pipenv run python -m spacy download en_core_web_sm

.PHONY: run
run:
	@pipenv run ./sentiment/cmd/main.py
