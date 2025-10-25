# save as export_tb_scalars.py and run: python export_tb_scalars.py /path/to/your/events.file out.csv
import sys, os, glob, pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

in_path  = sys.argv[1]          # a single events.* file OR a run dir containing them
out_csv  = sys.argv[2]

dir = os.path.dirname(in_path)

# Build a list of event files
files = []
if os.path.isdir(in_path):
    files = glob.glob(os.path.join(in_path, "events.*")) + glob.glob(os.path.join(in_path, "tfevents.*"))
else:
    files = [in_path]

rows = []
for f in files:
    ea = EventAccumulator(f); ea.Reload()
    for tag in ea.Tags().get("scalars", []):
        for s in ea.Scalars(tag):
            rows.append({
                "tag": tag,
                "step": s.step,
                "value": s.value,
                "wall_time": s.wall_time,
                "source_file": os.path.basename(f),
            })
pd.DataFrame(rows).to_csv(os.path.join(dir,out_csv), index=False)
print("Wrote", out_csv, "with", len(rows), "rows")
