from src.tracker_v2 import load_detections, run_tracker

JSON_FILE = "outputs/10G/envelope_detections.json"

detections = load_detections(JSON_FILE)
tracks = run_tracker(detections)

track_ids = [tr.track_id for tr in tracks]

print(f"Total tracks: {len(track_ids)}")
print(f"Unique IDs : {len(set(track_ids))}")

if len(track_ids) == len(set(track_ids)):
    print("PASS: All track IDs are globally unique.")
else:
    print("FAIL: Duplicate track IDs detected!")

    seen = set()
    for tid in track_ids:
        if tid in seen:
            print(f"Duplicate ID: {tid}")
        seen.add(tid)

print("\nPotential track handoffs:")
print("-" * 60)

for tr1 in tracks:
    for tr2 in tracks:

        if tr1.track_id == tr2.track_id:
            continue

        dt = tr2.frames[0] - tr1.frames[-1]

        if not (1 <= dt <= 2):
            continue

        dx = abs(tr2.xs[0] - tr1.xs[-1])
        dz = abs(tr2.zs[0] - tr1.zs[-1])

        if dx < 0.5 and dz < 1.0:
            print(
                f"{tr1.track_id:3d} -> {tr2.track_id:3d} | "
                f"Δt={dt:1d}  "
                f"Δx={dx:.3f}  "
                f"Δz={dz:.3f}"
            )