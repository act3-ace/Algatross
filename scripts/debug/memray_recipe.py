import memray

from algatross.utils.io import increment_filepath_maybe

tracker_path = increment_filepath_maybe("your_file_path.bin")
with memray.Tracker(tracker_path):
    # your code
    y = 2 + 2
