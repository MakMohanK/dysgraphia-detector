# serial_logger.py
import serial, argparse, csv, time, os
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--port', required=True, help='Serial port e.g. COM3 or /dev/ttyACM0')
parser.add_argument('--baud', default=115200, type=int)
parser.add_argument('--outdir', default='data', help='where to save CSVs')
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)
fname = datetime.now().strftime("session_%Y%m%d_%H%M%S.csv")
out_path = os.path.join(args.outdir, fname)

ser = serial.Serial(args.port, args.baud, timeout=1)
time.sleep(2)  # allow time for arduino reset
print("Logging to", out_path)

with open(out_path, 'w', newline='') as csvfile:
    writer = None
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line:
            continue
        # detect header
        if line.lower().startswith('timestamp') and writer is None:
            headers = [h.strip() for h in line.split(',')]
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            print("Header:", headers)
            continue
        if writer is None:
            # If no header from device, create basic header:
            headers = ['timestamp_ms','ax','ay','az','gx','gy','gz','p1','p2']
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
        vals = line.split(',')
        if len(vals) != len(writer.fieldnames):
            print("Skipping malformed:", line)
            continue
        row = dict(zip(writer.fieldnames, vals))
        writer.writerow(row)
        csvfile.flush()
        print(".", end="", flush=True)
