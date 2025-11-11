import cv2
import os
import numpy as np
import pyarrow.csv as pc
import pyarrow.compute as pc_compute
import pickle as pkl

# Path to your video file
video_path = '/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski/youtube_skijump_dataset/videos'


# Path to your CSV file
csv_file = "/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski/youtube_skijump_dataset/keypoints/train_new.csv"
# output_folder_poses = "/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski/youtube_skijump_dataset/keypoints/val_all.pkl"
output_folder = "/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski/youtube_skijump_dataset/images/Training"

# Read the CSV file into a pandas DataFrame
df = pc.read_csv(csv_file)


joints_name = ['head','rsho','relb','rhan','lsho','lelb','lhan','rhip','rkne','rank','lhip','lkne','lank','rsti','rsta','lsti','lsta']


video_rows = range(0,10)
# video_rows = [4,8,9]
video_poses = []
data = {}
data_all = {}

for video_id in video_rows:
    condition1 = pc_compute.equal(df['event'], video_id)

    filtered_table = df.filter(condition1)

    columns = filtered_table.column_names

    frame_num = np.array(filtered_table['frame_num']).tolist()
    joints_columns = columns[4:56]

    # poses = np.empty(shape=(len(joints_name),3))

    # data[str(video_id)] = {}

    # Open the video file
    cap = cv2.VideoCapture(os.path.join(video_path,'video_%s.mp4' %(video_id)))

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print( length )
    for f in frame_num:#range(70881,71881):
        break_condition = False
        condition2 = pc_compute.equal(df['frame_num'],f)
        # f_new = f-1
        # Combine the conditions using logical AND
        combined_condition = pc_compute.and_(condition1, condition2)
        joints_table = df.filter(combined_condition)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f-1)
        # Read the frame
        ret, frame = cap.read()
        output_file = os.path.join(output_folder, f"{video_id}_{f}.jpg")
        # if count in frame_num:
        #     print('Save frame: ', count)
             # save frame as JPEG file    
        for column_name in joints_columns:
            index = joints_name.index(column_name.split('_')[0])
        #     if '_x' in column_name:
        #         poses[index,0] = np.array(joints_table[column_name])
        #     elif '_y' in column_name:
        #         poses[index,1] = np.array(joints_table[column_name])
            if '_s' in column_name:
                if np.array(joints_table[column_name])[0] == 0:
                    break_condition = True

        if not break_condition:
            cv2.imwrite(output_file, frame)

        # video_poses.append(poses)

        # data[str(video_id)][str(f)] = poses

# with open(output_folder_poses, 'wb') as handle:
#     pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)

# data_all['gt_joints'] = np.array(video_poses)

# with open(output_folder_poses, 'wb') as handle:
#     pkl.dump(data_all, handle, protocol=pkl.HIGHEST_PROTOCOL)



print('here ok')

quit()
# Directory to save extracted frames
output_directory = "path/to/output/directory"
os.makedirs(output_directory, exist_ok=True)

# Annotated data: frame numbers where annotations are present
annotated_frames = [10, 20, 30, 40]  # Replace this with your annotated frame numbers

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()

# Iterate through annotated frames
for frame_number in annotated_frames:
    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    ret, frame = cap.read()

    # Check if frame was read successfully
    if not ret:
        print(f"Error: Unable to read frame {frame_number}")
        continue

    # Save the frame as an image
    output_file = os.path.join(output_directory, f"frame_{frame_number}.jpg")
    cv2.imwrite(output_file, frame)

    print(f"Frame {frame_number} saved as {output_file}")

# Release the video capture object
cap.release()