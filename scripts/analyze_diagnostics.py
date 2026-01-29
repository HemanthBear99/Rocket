import csv
import math
import numpy as np

FNAME = 'plots/diagnostics_full.csv'

cols = {}
rows = []
with open(FNAME, 'r', newline='') as fh:
    reader = csv.reader(fh)
    header = next(reader)
    cols = {name: i for i, name in enumerate(header)}
    for r in reader:
        rows.append(r)

n = len(rows)
if n == 0:
    print('No rows')
    raise SystemExit(1)

# Helper to get float column
def col(name):
    return np.array([float(rows[i][cols[name]]) for i in range(n)])

time = col('time')
pos_x = col('pos_x'); pos_y = col('pos_y'); pos_z = col('pos_z')
cmd_x = col('cmd_thrust_x'); cmd_y = col('cmd_thrust_y'); cmd_z = col('cmd_thrust_z')
actual_pitch = col('actual_pitch_deg')
flight_path = col('flight_path_angle_deg')
throttle = col('throttle')

# Compute local vertical and commanded angle from vertical
r = np.vstack([pos_x, pos_y, pos_z]).T
r_norm = np.linalg.norm(r, axis=1)
vertical = (r.T / r_norm).T
cmd = np.vstack([cmd_x, cmd_y, cmd_z]).T
cmd_norm = np.linalg.norm(cmd, axis=1)
# avoid zeros
cmd_unit = (cmd.T / np.where(cmd_norm==0,1,cmd_norm)).T

cos_cmd_vert = np.einsum('ij,ij->i', vertical, cmd_unit)
cos_cmd_vert = np.clip(cos_cmd_vert, -1.0, 1.0)
cmd_angle_deg = np.degrees(np.arccos(cos_cmd_vert))

# Difference: commanded angle-from-vertical minus actual body angle-from-vertical
diff = cmd_angle_deg - actual_pitch
absdiff = np.abs(diff)

# Summary stats
max_diff = np.max(absdiff)
max_idx = int(np.argmax(absdiff))
mean_diff = float(np.mean(absdiff))
median_diff = float(np.median(absdiff))

# Thresholds
thresholds = [5.0, 15.0, 30.0]
first_exceed = {}
for th in thresholds:
    idxs = np.where(absdiff > th)[0]
    first_exceed[th] = int(idxs[0]) if idxs.size>0 else None

print('Rows:', n)
print(f'Max |diff| = {max_diff:.2f} deg at t={time[max_idx]:.2f} s (cmd={cmd_angle_deg[max_idx]:.2f}°, actual={actual_pitch[max_idx]:.2f}°)')
print(f'Mean |diff| = {mean_diff:.2f}°, median = {median_diff:.2f}°')
for th in thresholds:
    idx = first_exceed[th]
    if idx is None:
        print(f'No samples exceed {th}°')
    else:
        print(f'First time |diff| > {th}° at t={time[idx]:.2f}s, |diff|={absdiff[idx]:.2f}°, cmd={cmd_angle_deg[idx]:.2f}°, actual={actual_pitch[idx]:.2f}°')

# Show a few sample times: launch (t<20), mid (40-100), later (>100)
samples = []
for start,end in [(0,20),(30,80),(100,200),(200,10000)]:
    idxs = np.where((time>=start)&(time<end))[0]
    if idxs.size>0:
        i = idxs[len(idxs)//2]
        samples.append(i)

print('\nSample rows (time, cmd_deg, actual_deg, diff_deg, throttle, phase approximation):')
for i in samples:
    print(f"t={time[i]:6.2f}s: cmd={cmd_angle_deg[i]:6.2f}°, actual={actual_pitch[i]:6.2f}°, diff={diff[i]:6.2f}°, |diff|={absdiff[i]:6.2f}°, throttle={throttle[i]:.2f}, flight_path={flight_path[i]:.2f}")

# Print top 5 largest diffs
order = np.argsort(-absdiff)
print('\nTop 5 largest mismatches:')
for k in range(5):
    i = int(order[k])
    print(f"#{k+1}: t={time[i]:6.2f}s, |diff|={absdiff[i]:6.2f}°, cmd={cmd_angle_deg[i]:6.2f}°, actual={actual_pitch[i]:6.2f}°, throttle={throttle[i]:.2f}")

# Save a small CSV of mismatches > 15 deg
out_rows = []
for i in np.where(absdiff>15.0)[0]:
    out_rows.append([time[i], cmd_angle_deg[i], actual_pitch[i], diff[i], absdiff[i], throttle[i]])
if out_rows:
    import csv
    with open('plots/mismatches_gt15.csv','w',newline='') as fh:
        w=csv.writer(fh)
        w.writerow(['time','cmd_deg','actual_deg','diff_deg','absdiff_deg','throttle'])
        w.writerows(out_rows)
    print('\nWrote plots/mismatches_gt15.csv with', len(out_rows), 'rows')
else:
    print('\nNo mismatches >15° found')
