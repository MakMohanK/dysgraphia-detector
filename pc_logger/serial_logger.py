# serial_logger.py
import argparse, csv, os, time
from datetime import datetime

import serial
from serial.tools import list_ports

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--port', required=True, help='Serial port e.g. COM11 or /dev/ttyACM0')
    p.add_argument('--baud', default=115200, type=int)
    p.add_argument('--outdir', default='data', help='Where to save CSVs')
    p.add_argument('--noheader', action='store_true',
                   help='Force default header even if a header line arrives')
    p.add_argument('--showports', action='store_true', help='List ports and exit')
    return p.parse_args()

def main():
    args = parse_args()

    if args.showports:
        print("Available ports:")
        for p in list_ports.comports():
            print(" ", p.device, "-", p.description)
        return

    os.makedirs(args.outdir, exist_ok=True)
    fname = datetime.now().strftime("session_%Y%m%d_%H%M%S.csv")
    out_path = os.path.join(args.outdir, fname)

    print(f"Opening {args.port} @ {args.baud} …")
    ser = serial.Serial(args.port, args.baud, timeout=1)
    time.sleep(2)  # allow MCU reset
    print("Logging to", out_path)

    writer = None
    using_default_header = False
    default_header = ['timestamp_ms','ax','ay','az','gx','gy','gz','pressure1','pressure2']

    # Make path absolute (helps with OneDrive / cwd confusion)
    out_path = os.path.abspath(out_path)
    print("Logging to", out_path)

    rows_written = 0
    malformed = 0
    rawlog_path = os.path.join(args.outdir, fname.replace(".csv", ".log"))
    rawfh = open(rawlog_path, 'w', newline='')

    with open(out_path, 'w', newline='') as csvfile:
        try:
            while True:
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode('utf-8', errors='ignore').strip()
                if not line:
                    continue

                # Keep a raw copy to debug formatting
                rawfh.write(line + "\n")

                # Determine header
                if writer is None:
                    if (not args.noheader) and line.lower().startswith('timestamp'):
                        headers = [h.strip() for h in line.split(',')]
                        writer = csv.DictWriter(csvfile, fieldnames=headers)
                        writer.writeheader()
                        csvfile.flush(); os.fsync(csvfile.fileno())
                        print("Header from device:", headers)
                        continue
                    else:
                        writer = csv.DictWriter(csvfile, fieldnames=default_header)
                        writer.writeheader()
                        csvfile.flush(); os.fsync(csvfile.fileno())
                        using_default_header = True

                vals = [v.strip() for v in line.strip().strip(',').split(',')]
                if using_default_header:
                    t_ms = int(time.time() * 1000)
                    vals = [str(t_ms)] + vals

                if len(vals) != len(writer.fieldnames):
                    malformed += 1
                    print("Skipping malformed (", len(vals), "vs", len(writer.fieldnames), "):", line)
                    continue

                row = dict(zip(writer.fieldnames, vals))
                writer.writerow(row)
                rows_written += 1

                # hard flush so you can see the file grow
                csvfile.flush(); os.fsync(csvfile.fileno())

                if rows_written % 50 == 0:
                    print(f"Wrote {rows_written} rows (malformed: {malformed})", flush=True)
        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            try: ser.close()
            except: pass
            rawfh.close()
            print(f"Closed port. CSV saved to: {out_path}")
            print(f"Raw log saved to: {rawlog_path}")

if __name__ == "__main__":
    main()
