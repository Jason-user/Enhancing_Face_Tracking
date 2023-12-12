# Meta Face Tracking

## Tracking
![](gif_file/avtdrvnfyh_tracking.gif)

## Meta Tracking
![](gif_file/avtdrvnfyh_meta.gif)

## Environment setup
```
git clone https://github.com/Jason-user/Meta_tracking
```
```
conda create --name your_env_name python=3.10
```
```
conda activate your_env_name
```
```
cd Meta_tracking
```
```
conda env create -f environment.yml
```
Install the modified version of facexlib by
```
cd facexlib
pip install -e .
```

## Download pre-trained weight
After downloading [resnet18](https://github.com/Jason-user/Meta_tracking/releases), put it into the folder


## Tracking
Run the inference script by (Single GPU)
```
python ego4d_face_tracking.py --input_folder (your_video_path) --save_folder (output_dir)
```
Multi GPU (You can refer to [Accelerate](https://huggingface.co/docs/accelerate/basic_tutorials/notebook))

```
accelerate config
accelerate launch ego4d_face_tracking.py --input_folder (your_video_path) --save_folder (output_dir)
```

## Meta Tracking
Put videos and csv_files (generated from ego4d_face_tracking.py) together (Within same folder). \
then run:
```
python meta_tracking.py --old_csv_video_path (csv_file and video path) --filename (The video you are going to process)
```

If the same face is identified with different IDs after meta_tracking, you can increase the "distance_threshold" in meta_tracking.py, at row number 158

## Demo
After generating the new_csv_file (from meta_tracking.py), put it and the video together(within same folder)
```
python demo.py --csv_video_path (csv_file and video path) --video_name (The video you are going to process)
```