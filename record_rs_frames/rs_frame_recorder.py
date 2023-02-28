import pyrealsense2 as rs
import time
import cv2
import os
import numpy as np
import json
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

def line_select_callback(eclick, erelease):
    global CURRENT_ROI
    'eclick and erelease are the press and release events'
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))

    CURRENT_ROI = [float(eclick.xdata),
        float(eclick.ydata),
        float(erelease.xdata),
        float(erelease.ydata)
    ]

    plt.close('all')


def toggle_selector(event):
    print(' Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)


#### parameters ####
ITEM_NAME = 'Pineapple_1'
ITEM_NUMBER = '035'
FRAME_COUNT_OFFSET = 1000

NUMBER_OF_FRAMES = 1000
CURRENT_ROI = [0, 0, 640, 480]

SAVE_TO_PATH = '/media/michel_ma/NVMe2/Paper_Dataset/' + ITEM_NUMBER + "_" + ITEM_NAME
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

frames = pipeline.wait_for_frames()
frames = align.process(frames)
color_frame = frames.get_color_frame()
color_image = np.asanyarray(color_frame.get_data())

plt.imshow(color_image)

current_ax = plt.gca()

toggle_selector.RS = RectangleSelector(current_ax, line_select_callback,
                                       drawtype='box', useblit=True,
                                       button=[1, 3],  # don't use middle button
                                       minspanx=5, minspany=5,
                                       spancoords='pixels',
                                       interactive=True)

plt.connect('key_press_event', toggle_selector)
plt.show()

roi_path = os.path.join(SAVE_TO_PATH, 'roi.json' )
dictionary = {}

for i in range(-SECONDS_TO_CAPTURE, NUMBER_OF_FRAMES):

    if not os.path.exists(SAVE_TO_PATH):
        os.makedirs(SAVE_TO_PATH)

    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
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
        depth_name = os.path.join(SAVE_TO_PATH, ITEM_NAME +"_"+ str(FRAME_COUNT_OFFSET + i).zfill(6) + '.depth.png')
        cv2.imwrite(depth_name, depth_image)
    if color_frame:
        color_name = os.path.join(SAVE_TO_PATH, ITEM_NAME +"_"+ str(FRAME_COUNT_OFFSET + i).zfill(6) + '.png')
        cv2.imwrite(color_name, color_image)

    current_image = ITEM_NAME +"_"+ str(FRAME_COUNT_OFFSET + i).zfill(6)
    dictionary[current_image] = CURRENT_ROI

    time.sleep(0.2)

json_object = json.dumps(dictionary, indent=0)

with open(roi_path, "a") as outfile:
    outfile.write(json_object)

print("Done")

    

    