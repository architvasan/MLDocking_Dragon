.DEFAULT_GOAL := all
package_name = pipt
extra_folders = 
isort = isort $(package_name) $(extra_folders)
black = black --target-version py37 $(package_name) $(extra_folders)
flake8 = flake8 $(package_name)/ $(extra_folders)
pylint = pylint $(package_name)/ $(extra_folders)
pydocstyle = pydocstyle $(package_name)/
run_mypy = mypy --config-file setup.cfg

.PHONY: install
install:
	pip install -U pip setuptools wheel
	pip install -r requirements/dev.txt
	pip install -r requirements/requirements.txt
	pip install -e .

.PHONY: format
format:
	$(isort)
	$(black)

.PHONY: lint
lint:
	$(black) --check --diff
	$(flake8)
	#$(pylint)
	#$(pydocstyle)

.PHONY: mypy
mypy:
	$(run_mypy) --package $(package_name)
	$(run_mypy) $(package_name)/
	$(run_mypy) $(extra_folders)

.PHONY: coverage
coverage:
	python -m coverage run -m pytest tests
	python -m coverage report
	python -m coverage html
	open htmlcov/index.html

.PHONY: pygount
pygount:
	pygount --format=summary $(package_name)

.PHONY: radon
radon:
	radon cc $(package_name) -a

.PHONY: all
all: format lint #mypy