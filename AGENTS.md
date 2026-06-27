# Project Guidelines

## General
- This package provides an API for the *Graph Attribute and Structure Matching* ([GASM](https://jgaa.info/index.php/jgaa/article/view/2979)) algorithm, on both CPU and GPU.

## AGENTS.md
- Update this file when needed to keep instructions clear and actionable.
- If a sentence is unclear, rewrite it with precise project terms.

## Environment
- Use only the Python virtual environment located at /home/raphael/Science/Projects/.virtual_environments/GASM.
- Prefer this interpreter for running scripts, tests, and analysis commands.

## Architecture
- This repository is composed of the folloxing folders:
  - gasm: source code
  - doc: documentation
  - benchmark: tests and benchmarks

## Conventions
- Preserve the existing scientific and numerical scripting style unless a change is necessary.
- In particular, keep the notations of the original article (link above) as much as possible.

## Runtime Notes
- The GPU implementation depends on OpenCL availability on the host system.
- If OpenCL is unavailable, report the runtime limitation rather than changing the logic to hide it.
  
## Documentation

### Aim and scope
- The documentation is targeting two audiences: users and developpers.

### Technical notes
- The documentation is generated with sphinx. Use `sphinx-build doc/source/ doc/build` to build the documentation.
- Before every documentation building, check that the version number (release) in doc/source/conf.py is incremented with respect to the version number (VERSION) in setup.py.
- Build the documentation every time you make modifications to the source folder.
- All references to modules, classes, methods or functions in the documentation should be hyperlinks pointing to the corresponding element in the API reference.