all: docs

.PHONY: clean
clean:
	find . -name "*.pyc" -delete

.PHONY: docs
docs:
	sphinx-apidoc -f -o docs/source pipeline && cd docs && make html && cd ..
