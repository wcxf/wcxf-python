# WCxf Python API and command line interface

This Python package provides a Python API and command line interface to
perform the following operations on WCxf files (or dictionaries):

- *validation* against EFT and basis files,
- *conversion* between JSON and YAML,
- *translation* between different bases, and
- *matching* between different EFTs.

## Installation

Since the package is currently in active development, the preferred way
to install it is in development mode. From the base directory:

```bash
pip3 install -e . --user
```

## Command line interface

The CLI provides the following commands:

### convert

```
usage: wcxf-convert [-h] [--output [OUTPUT]] FORMAT [FILE]

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
usage: wcxf-translate [-h] [--output [OUTPUT]] [--format FORMAT] BASIS [FILE]

Command line script for basis translation of WCxf files.

positional arguments:
  BASIS              Output basis
  FILE               Input file. If "-", read from standard input

optional arguments:
  -h, --help         show this help message and exit
  --output [OUTPUT]  Output file. If absent, print to standard output
  --format FORMAT    Output format (default: json)
  ```
