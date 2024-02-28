import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datetime

pipe = rs.pipeline()
cfg = rs.config()

align = rs.align(rs.stream.color)

cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipe.start(cfg)

i = 0
# get the date and time into format string
date_time = datetime.datetime.now().strftime("%Y%m%d")

try:
    frame_count = [int(np.loadtxt(f"./data/{date_time}/count.txt"))]
except:
    frame_count = [0]

print("frame count: ", frame_count)

try:
    os.makedirs(f"./data/{date_time}/frames")
    os.makedirs(f"./data/{date_time}/data")
except:
    pass

while True:
    frameset = pipe.wait_for_frames()
    frame = align.process(frameset)

    depth_frame = frame.get_depth_frame()
    color_frame = frame.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    greyscale = cv2.convertScaleAbs(depth_image, alpha=0.07)

    cv2.imshow('rgb', color_image)
    cv2.imshow('gray', greyscale)

    if i % 5 == 0:
        np.save(f"./data/{date_time}/data/data_{frame_count[0]:04d}.npy", greyscale)
        cv2.imwrite(f"./data/{date_time}/frames/frame_{frame_count[0]:04d}.png", color_image)
        frame_count[0] += 1
        print("exporting frame", frame_count[0])

    i += 1

    if cv2.waitKey(1) == ord('q'):
        break

pipe.stop()


np.savetxt(f"./data/{date_time}/count.txt", frame_count)
