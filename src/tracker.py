from dataclasses import dataclass, field
import json
from collections import defaultdict
from matplotlib.pylab import det
import numpy as np
from scipy.optimize import linear_sum_assignment

@dataclass
class Track:
    """
    One reconstructed spicule trajectory.
    """

    track_id: int

    frames: list = field(default_factory=list)
    times: list = field(default_factory=list)
    xs: list = field(default_factory=list)
    zs: list = field(default_factory=list)

    missed_frames: int = 0
    active: bool = True

    def add_detection(self, frame, time, x, z):
        self.frames.append(frame)
        self.times.append(time)
        self.xs.append(x)
        self.zs.append(z)
        self.missed_frames = 0

    def miss(self):
        self.missed_frames += 1

    @property
    def last_x(self):
        return self.xs[-1]

    @property
    def last_z(self):
        return self.zs[-1]

    @property
    def last_frame(self):
        return self.frames[-1]
    
    @property
    def predicted_x(self):
        if len(self.xs) < 2:
            return self.last_x
        return self.xs[-1] + (self.xs[-1] - self.xs[-2])


    @property
    def predicted_z(self):
        if len(self.zs) < 2:
            return self.last_z
        return self.zs[-1] + (self.zs[-1] - self.zs[-2])

def load_detections(json_file):
    """
    Load envelope detections and group them by frame.

    Returns
    -------
    dict:
        frame_number -> list of detections
    """

    with open(json_file, "r") as f:
        data = json.load(f)

    detections = defaultdict(list)

    for d in data:
        detections[d["frame"]].append(d)

    return detections


def build_cost_matrix(
    tracks,
    detections,
    sigma_x=0.5,
    sigma_z=1.0,
):
    """
    Build assignment cost matrix.
    """

    C = np.zeros((len(tracks), len(detections)))
    MAX_DX = 0.8   # Mm
    MAX_DZ = 2.0   # Mm

    for i, tr in enumerate(tracks):
        for j, det in enumerate(detections):

            # dx = (det["x"] - tr.predicted_x) / sigma_x
            # dz = (det["z"] - tr.predicted_z) / sigma_z
            BIG = 1e6

            dx_phys = abs(det["x"] - tr.predicted_x)
            dz_phys = abs(det["z"] - tr.predicted_z)

            if dx_phys > MAX_DX or dz_phys > MAX_DZ:
                C[i, j] = BIG
            else:
                dx = dx_phys / sigma_x
                dz = dz_phys / sigma_z
                C[i, j] = np.sqrt(dx*dx + dz*dz)

            # C[i, j] = np.sqrt(dx * dx + dz * dz)

    return C

def run_tracker(
    detections_by_frame,
    max_cost=3.0,
    max_missed=2,
):
    """
    Build tracks from frame-by-frame detections.
    """

    tracks = []
    next_track_id = 0

    frames = sorted(detections_by_frame.keys())

    for frame in frames:

        current = detections_by_frame[frame]

        active_tracks = [
            tr for tr in tracks
            if tr.active
        ]

        # -------------------------------------------------
        # Bootstrap first frame
        # -------------------------------------------------

        if len(active_tracks) == 0:

            for det in current:
                tr = Track(next_track_id)
                tr.add_detection(
                    det["frame"],
                    det["time"],
                    det["x"],
                    det["z"],
                )
                tracks.append(tr)
                next_track_id += 1

            continue

        # -------------------------------------------------
        # Assignment
        # -------------------------------------------------

        cost = build_cost_matrix(
            active_tracks,
            current,
        )

        row_ind, col_ind = linear_sum_assignment(cost)

        assigned_tracks = set()
        assigned_dets = set()

        for r, c in zip(row_ind, col_ind):

            if cost[r, c] > max_cost:
                continue

            tr = active_tracks[r]
            det = current[c]

            tr.add_detection(
                det["frame"],
                det["time"],
                det["x"],
                det["z"],
            )

            assigned_tracks.add(r)
            assigned_dets.add(c)

        # -------------------------------------------------
        # Handle unmatched tracks
        # -------------------------------------------------

        for i, tr in enumerate(active_tracks):
            if i not in assigned_tracks:
                tr.miss()

                if tr.missed_frames > max_missed:
                    tr.active = False

        # -------------------------------------------------
        # Handle unmatched detections
        # -------------------------------------------------

        for j, det in enumerate(current):
            if j not in assigned_dets:

                tr = Track(next_track_id)
                tr.add_detection(
                    det["frame"],
                    det["time"],
                    det["x"],
                    det["z"],
                )

                tracks.append(tr)
                next_track_id += 1

    return tracks