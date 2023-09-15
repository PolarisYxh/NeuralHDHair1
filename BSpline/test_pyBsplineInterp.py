import struct
import numpy as np
import os
import pyBsplineInterp
def readhair(file_name):
    with open(file_name, mode='rb')as f:
        num_strand = f.read(4)
        (num_strand,) = struct.unpack('I', num_strand)
        point_count = f.read(4)
        (point_count,) = struct.unpack('I', point_count)

        # print("num_strand:",num_strand)
        segments = f.read(2 * num_strand)
        segments = struct.unpack('H' * num_strand, segments)
        segments = list(segments)
        num_points = sum(segments)

        points = f.read(4 * num_points * 3)
        points = struct.unpack('f' * num_points * 3, points)
        segments = np.array(segments)
        points = np.array(points).reshape((-1,3))
    return segments,points
segments,points = readhair("hair_delete.hair")
x=points[:96+90]
# print(points[:96])
# print(segments[:2])
import time
start = time.time()
points1=pyBsplineInterp.GetBsplineInterp(points,segments,100,3)
end = time.time()
response_time = end - start
pass
# points = process_list(points,segments,self.sample_num)