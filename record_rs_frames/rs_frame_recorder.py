import pyrealsense2 as rs
import time
import cv2
import os
import numpy as np

#### parameters ####
ITEM_NAME = 'Small_Angle_Grinder_real'
ITEM_NUMBER = '999'
FRAME_COUNT_OFFSET = 0

NUMBER_OF_FRAMES = 20

SAVE_TO_PATH = '/media/michel_ma/NVMe2/MA_Heinemann_Dataset/003_Test/002_Unknown/' + ITEM_NUMBER + "_" + ITEM_NAME
COLOR_RES = [640, 480]
DEPTH_RES = [640, 480]

SECONDS_TO_CAPTURE = 5

#### parameters end ####

pipeline = rs.pipeline()
config = rs.config()
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)

config.enable_stream(rs.stream.depth, DEPTH_RES[0], DEPTH_RES[1], rs.format.z16, 30)
config.enable_stream(rs.stream.color, COLOR_RES[0], COLOR_RES[1], rs.format.bgr8, 30)
align = rs.align(rs.stream.color)

cfg = pipeline.start(config)
time.sleep(2)

profile_1 = cfg.get_stream(rs.stream.depth) 
intr_depth = profile_1.as_video_stream_profile().get_intrinsics()

profile_2 = cfg.get_stream(rs.stream.color)
intr_color = profile_2.as_video_stream_profile().get_intrinsics()
extr = profile_1.get_extrinsics_to(profile_1)

print(intr_depth)
print(intr_color)
print(extr)

for i in range(-SECONDS_TO_CAPTURE, NUMBER_OF_FRAMES):

    if not os.path.exists(SAVE_TO_PATH):
        os.makedirs(SAVE_TO_PATH)

    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data()) * 10
    color_image = np.asanyarray(color_frame.get_data())

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    cv2.imshow('Stream', color_image)
    if cv2.waitKey(1) == ord("q"):
        break

    if i<0:
        print("Recording in " + str(np.abs(i)) + " seconds")
        time.sleep(1)
        continue
    else:
        print("Recording frame: " + str(np.abs(FRAME_COUNT_OFFSET + i)))

    if depth_frame:
        depth_name = os.path.join(SAVE_TO_PATH, ITEM_NAME + str(FRAME_COUNT_OFFSET + i).zfill(6) + '.depth.png')
        cv2.imwrite(depth_name, depth_image)
    if color_frame:
        color_name = os.path.join(SAVE_TO_PATH, ITEM_NAME + str(FRAME_COUNT_OFFSET + i).zfill(6) + '.png')
        cv2.imwrite(color_name, color_image)
    time.sleep(0.2)

print("Done")

    

    