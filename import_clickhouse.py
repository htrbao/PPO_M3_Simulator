import torch
import argparse
import json
import time
import os
import pandas as pd
from datetime import datetime
from test.database import ClickhouseDB
from test.global_config import DB_CONFIG
from test.utils import import_single_model

def get_arguments():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--report_date', type=str)
    parser.add_argument('--checkpoint', type=str)
    
    return parser.parse_args()


def import_models(store_dir, report_date, db):
    row = 0
    for d in os.listdir(store_dir):
        checkpoint_path = os.path.join(store_dir, d)
        if os.path.isdir(checkpoint_path):
            row += import_single_model(checkpoint_path, report_date, db)
    return row
    
def main():
    args = get_arguments()
    store_dir = "_saved_test"
    report_date = datetime.strptime(args.report_date, "%Y-%m-%d")
    db = ClickhouseDB(DB_CONFIG["host"], DB_CONFIG["port"], DB_CONFIG["user"], DB_CONFIG["password"], DB_CONFIG["database"])
    if args.checkpoint:
        checkpoint_path = os.path.join(store_dir, os.path.basename(args.checkpoint).split(".")[0])
        row = import_single_model(checkpoint_path, report_date, db)
    else:
        row = import_models(store_dir, report_date, db)

    print(f"Import successfully! {row}")
    
if __name__ == '__main__':
    main()
