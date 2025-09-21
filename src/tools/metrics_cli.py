#!/usr/bin/env python3
import argparse
import csv
import math
import statistics

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def load_rows(path):
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


def compute_metrics(rows):
    xs = [float(r['x']) for r in rows]
    ys = [float(r['y']) for r in rows]
    conf = [float(r['vps_conf']) for r in rows]
    ofc = [float(r['of_conf']) for r in rows]
    fps_est = []
    ts = [float(r['t']) for r in rows]
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i-1]
        if dt > 0:
            fps_est.append(1.0/dt)
    metrics = {
        'num_rows': len(rows),
        'x_span_m': (min(xs), max(xs)),
        'y_span_m': (min(ys), max(ys)),
        'vps_conf_mean': statistics.mean(conf) if conf else 0.0,
        'of_conf_mean': statistics.mean(ofc) if ofc else 0.0,
        'fps_mean': statistics.mean(fps_est) if fps_est else 0.0,
    }
    # Position error if ENU truth available
    try:
        tx = [float(r['truth_x']) for r in rows]
        ty = [float(r['truth_y']) for r in rows]
        tz = [float(r['truth_z']) for r in rows]
        errs = []
        for x, y, z, gx, gy, gz in zip(xs, ys, [0.0]*len(xs), tx, ty, tz):
            if not (math.isnan(gx) or math.isnan(gy) or math.isnan(gz)):
                dx = x - gx
                dy = y - gy
                # If z is modeled, include; else skip
                dz = 0.0 if math.isnan(gz) else (0.0)  # placeholder; height not modeled elsewhere
                errs.append(math.sqrt(dx*dx + dy*dy + dz*dz))
        if errs:
            metrics['pos_rmse_m'] = math.sqrt(sum(e*e for e in errs) / len(errs))
    except Exception:
        pass
    # If truth available, compute naive yaw error
    yaw = [float(r['yaw']) for r in rows]
    try:
        yaw_truth = [float(r['truth_yaw']) for r in rows]
        yaw_err = [abs(a - b) for a, b in zip(yaw, yaw_truth) if not math.isnan(b)]
        if yaw_err:
            metrics['yaw_err_mean_rad'] = statistics.mean(yaw_err)
    except Exception:
        pass
    return metrics


def maybe_plot(rows):
    if plt is None:
        return
    xs = [float(r['x']) for r in rows]
    ys = [float(r['y']) for r in rows]
    conf = [float(r['vps_conf']) for r in rows]
    plt.figure()
    sc = plt.scatter(xs, ys, c=conf, cmap='viridis', s=6)
    plt.colorbar(sc, label='VPS confidence')
    plt.title('Trajectory (colored by VPS confidence)')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='Path to eval CSV')
    ap.add_argument('--plot', action='store_true', help='Show simple plots')
    args = ap.parse_args()

    rows = load_rows(args.csv)
    m = compute_metrics(rows)
    for k, v in m.items():
        print(f"{k}: {v}")
    if args.plot:
        maybe_plot(rows)


if __name__ == '__main__':
    main() 