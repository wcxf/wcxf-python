[![Build Status](https://travis-ci.org/wcxf/wcxf-python.svg?branch=master)](https://travis-ci.org/wcxf/wcxf-python) [![Coverage Status](https://coveralls.io/repos/github/wcxf/wcxf-python/badge.svg?branch=master)](https://coveralls.io/github/wcxf/wcxf-python?branch=master)

# WCxf Python API and command line interface

This Python package provides a Python API and command line interface to
perform the following operations on WCxf files (or dictionaries):

- *validation* against EFT and basis files,
- *conversion* between JSON and YAML,
- *translation* between different bases, and
- *matching* between different EFTs.

## Installation

```bash
pip3 install wcxf
```

## Command line interface

The CLI provides the following commands:

### convert

```
usage: wcxf convert [-h] [--output [OUTPUT]] FORMAT [FILE]

Command line script to convert WCxf files between YAML and JSON.

positional arguments:
  FORMAT             Output format (should be yaml or json)
  FILE               Input file. If "-", read from standard input

optional arguments:
  -h, --help         show this help message and exit
  --output [OUTPUT]  Output file. If absent, print to standard output
  ```

### translate

```
usage: wcxf translate [-h] [--output [OUTPUT]] [--format FORMAT] BASIS [FILE]

Command line script for basis translation of WCxf files.

positional arguments:
  BASIS              Output basis
  FILE               Input file. If "-", read from standard input

optional arguments:
  -h, --help         show this help message and exit
  --output [OUTPUT]  Output file. If absent, print to standard output
  --format FORMAT    Output format (default: json)
  ```

### match

```
usage: wcxf match [-h] [--output [OUTPUT]] [--format FORMAT] EFT BASIS [FILE]

Command line script for matching of WCxf files.

positional arguments:
  EFT                Output EFT
  BASIS              Output basis
  FILE               Input file. If "-", read from standard input

optional arguments:
  -h, --help         show this help message and exit
  --output [OUTPUT]  Output file. If absent, print to standard output
  --format FORMAT    Output format (default: json)
```

### validate

```
usage: wcxf validate [-h] TYPE [FILE]

Command line script for validation of WCxf files.

positional arguments:
  TYPE        Type of file to validate: should be 'eft', 'basis', or 'wc'
  FILE        Input file. If "-", read from standard input

optional arguments:
  -h, --help  show this help message and exit
```
