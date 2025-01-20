PYVER:=$(shell python3 -c 'from sys import version_info as v; print("{0}.{1}".format(v[0], v[1]))')
PYTHON = python$(PYVER)

# You can set these variables from the command line, and also
# from the environment for the first four.
SPHINXOPTS    ?= -nT
SPHINXBUILD   ?= LANG=C sphinx-build
DOCSSOURCEDIR	    = docs/source
DOCSBUILDDIR	   	= docs/build

HUGOSOURCEDIR		= $(DOCSSOURCEDIR)/hugo
HUGOBUILDDIR		= $(DOCSBUILDDIR)/public
HUGOAPIDIR			= $(HUGOSOURCEDIR)/static/docs

SPHINXSOURCEDIR     = $(DOCSSOURCEDIR)/sphinx
SPHINXBUILDDIR      = $(DOCSBUILDDIR)/sphinx
SPHINXAPIGENDIR		= $(SPHINXSOURCEDIR)/docs

# SEARCH = (echo "Installing \`pagefind\` and generating search index..." && npx --yes pagefind --site $(HUGOBUILDDIR))
SEARCH = (echo "Installing \`pagefind\` and generating search index..." && npx --yes pagefind)

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SPHINXSOURCEDIR)" "$(SPHINXBUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

autogen-modules:
	$(PYTHON) $(SPHINXSOURCEDIR)/preprocess.py

build:
	@hugo
	@$(SEARCH)

serve: build
	@hugo --printI18nWarnings server

serve-dev: build
	@hugo --printI18nWarnings server --buildDrafts --disableFastRender --poll 1000ms

clean:
	@rm -rf $(HUGOBUILDDIR)/*
	@rm -rf $(SPHINXBUILDDIR)/*
	@rm -rf $(SPHINXAPIGENDIR)/*
	@rm -rf $(SPHINXSOURCEDIR)/generated/*
	@rm -rf $(HUGOAPIDIR)/*

clean-%: clean
	@make "$(subst clean-,,$@)"
# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile autogen-modules
	@$(SPHINXBUILD) -M "$(subst clean-,,$@)" "$(SPHINXSOURCEDIR)" "$(SPHINXBUILDDIR)" "$(SPHINXOPTS)" $(O)
	@mkdir -p "$(HUGOAPIDIR)"
	@cp -rf "$(SPHINXBUILDDIR)"/"$(subst clean-,,$@)"/* "$(HUGOAPIDIR)"
	@make build
