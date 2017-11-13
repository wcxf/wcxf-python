import argparse
import wcxf
import yaml
import json
import sys
import logging

def convert_json(stream_in, stream_out):
    try:
        return wcxf.classes._yaml_to_json(stream_in, stream_out, indent=2)
    except yaml.YAMLError:
        logging.error("Input file cannot be parsed as YAML.")
        return 1

def convert_yaml(stream_in, stream_out):
    try:
        return wcxf.classes._json_to_yaml(stream_in, stream_out, default_flow_style=False)
    except json.decoder.JSONDecodeError:
        logging.error("Input file cannot be parsed as JSON.")
        return 1

def convert():
    parser = argparse.ArgumentParser(description="""Command line script to convert WCxf files between YAML and JSON.""",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("FORMAT", help="Output format (should be yaml or json)", type=str)
    parser.add_argument("FILE", nargs='?', help="Input file. If \"-\", read from standard input",
                        type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument("--output", nargs='?', help="Output file. If absent, print to standard output",
                        type=argparse.FileType('w'), default=sys.stdout)
    args = parser.parse_args()
    if args.FORMAT.lower() == 'json':
        convert_json(args.FILE, args.output)
    if args.FORMAT.lower() == 'yaml':
        convert_yaml(args.FILE, args.output)

def translate():
    parser = argparse.ArgumentParser(description="""Command line script for basis translation of WCxf files.""",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("BASIS", help="Output basis", type=str)
    parser.add_argument("FILE", nargs='?', help="Input file. If \"-\", read from standard input",
                        type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument("--output", nargs='?', help="Output file. If absent, print to standard output",
                        type=argparse.FileType('w'), default=sys.stdout)
    parser.add_argument("--format", help="Output format (default: json)", type=str, default="json")
    args = parser.parse_args()
    wc_in = wcxf.WC.load(args.FILE)
    wc_out = wc_in.translate(args.BASIS)
    wc_out.dump(stream=args.output, fmt=args.format)

def match():
    parser = argparse.ArgumentParser(description="""Command line script for matching of WCxf files.""",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("EFT", help="Output EFT", type=str)
    parser.add_argument("BASIS", help="Output basis", type=str)
    parser.add_argument("FILE", nargs='?', help="Input file. If \"-\", read from standard input",
                        type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument("--output", nargs='?', help="Output file. If absent, print to standard output",
                        type=argparse.FileType('w'), default=sys.stdout)
    parser.add_argument("--format", help="Output format (default: json)", type=str, default="json")
    args = parser.parse_args()
    wc_in = wcxf.WC.load(args.FILE)
    wc_out = wc_in.match(args.EFT, args.BASIS)
    wc_out.dump(stream=args.output, fmt=args.format)

def validate():
    parser = argparse.ArgumentParser(description="""Command line script for validation of WCxf files.""",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("TYPE", help="Type of file to validate: should be 'eft', 'basis', or 'wc'", type=str)
    parser.add_argument("FILE", nargs='?', help="Input file. If \"-\", read from standard input",
                        type=argparse.FileType('r'), default=sys.stdin)
    args = parser.parse_args()
    if args.TYPE == 'eft':
        eft = wcxf.EFT.load(args.FILE)
    elif args.TYPE == 'basis':
        basis = wcxf.Basis.load(args.FILE)
        basis.validate()
    elif args.TYPE == 'wc':
        wc = wcxf.WC.load(args.FILE)
        wc.validate()
    else:
        logging.error("TYPE should be 'eft', 'basis', or 'wc'")
        return 1
    print("Validation successful.")
    return 0
