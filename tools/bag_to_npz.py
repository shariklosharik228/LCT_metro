#!/usr/bin/env python3
"""Read PointCloud2 frames straight from a ROS 2 .db3 bag (no ROS needed).

Only the standard library (sqlite3) and NumPy are required, so it runs on any
machine.  Frames are stored compactly (xyz in centimetres, int16) together
with their header stamps:

    python3 tools/bag_to_npz.py bag_dir_or_db3 out.npz [--every 1] [--max-range 330]
"""
import argparse
import glob
import os
import sqlite3
import struct

import numpy as np

_DT = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}


class _Cdr:
    def __init__(self, buf):
        self.b = buf
        self.p = 4                      # skip encapsulation header
        self.e = '<' if buf[1] == 1 else '>'

    def align(self, n):
        off = (self.p - 4) % n
        if off:
            self.p += n - off

    def u32(self):
        self.align(4)
        v = struct.unpack_from(self.e + 'I', self.b, self.p)[0]
        self.p += 4
        return v

    def i32(self):
        self.align(4)
        v = struct.unpack_from(self.e + 'i', self.b, self.p)[0]
        self.p += 4
        return v

    def u8(self):
        v = self.b[self.p]
        self.p += 1
        return v

    def string(self):
        n = self.u32()
        s = bytes(self.b[self.p:self.p + n - 1]).decode()
        self.p += n
        return s


def parse_pointcloud2(blob):
    r = _Cdr(blob)
    sec, nsec = r.i32(), r.u32()
    frame = r.string()
    height, width = r.u32(), r.u32()
    fields = {}
    for _ in range(r.u32()):
        name = r.string()
        offset = r.u32()
        dtype = r.u8()
        r.u32()                         # count
        fields[name] = (offset, dtype)
    big = bool(r.u8())
    point_step, row_step = r.u32(), r.u32()
    n = r.u32()
    data = np.frombuffer(blob, np.uint8, count=n, offset=r.p)
    end = '>' if big else '<'
    dt = np.dtype({'names': ['x', 'y', 'z'],
                   'formats': [end + _DT[fields[k][1]] for k in 'xyz'],
                   'offsets': [fields[k][0] for k in 'xyz'], 'itemsize': point_step})
    cloud = np.frombuffer(data.tobytes(), dt, count=height * width)
    xyz = np.column_stack((cloud['x'], cloud['y'], cloud['z'])).astype(np.float32)
    return sec + nsec * 1e-9, frame, xyz


def iter_bag(path, topic=None):
    db = path if path.endswith('.db3') else sorted(glob.glob(os.path.join(path, '*.db3')))[0]
    con = sqlite3.connect(db)
    topics = {tid: (name, typ) for tid, name, typ in
              con.execute('SELECT id, name, type FROM topics')}
    ids = [tid for tid, (name, typ) in topics.items()
           if typ == 'sensor_msgs/msg/PointCloud2' and (topic is None or name == topic)]
    q = 'SELECT timestamp, data FROM messages WHERE topic_id IN (%s) ORDER BY timestamp' % \
        ','.join(str(i) for i in ids)
    for ts, blob in con.execute(q):
        yield (ts,) + parse_pointcloud2(blob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('out')
    ap.add_argument('--topic')
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--max-range', type=float, default=330.0)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--count', type=int, default=10 ** 9)
    a = ap.parse_args()
    stamps, sizes, chunks = [], [], []
    frame_id = ''
    axis = sign = None
    for i, (ts, stamp, frame_id, xyz) in enumerate(iter_bag(a.bag, a.topic)):
        if i < a.start or (i - a.start) % a.every:
            continue
        if len(stamps) >= a.count:
            break
        xyz = xyz[np.isfinite(xyz).all(1)]
        if axis is None:   # forward = direction of the far returns (tunnel axis)
            r = np.linalg.norm(xyz, axis=1)
            far = xyz[r > 60]
            dirn = (far / np.linalg.norm(far, axis=1)[:, None]).mean(0)
            axis, sign = int(np.argmax(np.abs(dirn[:2]))), float(np.sign(dirn[np.argmax(np.abs(dirn[:2]))]))
            print('forward axis', 'xy'[axis], sign, flush=True)
        f = sign * xyz[:, axis]
        xyz = xyz[(f > 1.0) & (f < a.max_range) & (np.abs(xyz[:, 1 - axis]) < 40)]
        chunks.append(np.round(xyz * 100).astype(np.int32).clip(-32767, 32767).astype(np.int16))
        stamps.append(stamp)
        sizes.append(len(xyz))
        if len(stamps) % 20 == 0:
            print(len(stamps), 'frames', flush=True)
    np.savez(a.out, xyz_cm=np.vstack(chunks), sizes=np.asarray(sizes), stamps=np.asarray(stamps),
             frame_id=frame_id)
    print('saved', len(stamps), 'frames, points/frame median', int(np.median(sizes)))


if __name__ == '__main__':
    main()
