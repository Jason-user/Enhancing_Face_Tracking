# Meta-Face-Tracking
## After Tracking
![](avtdrvnfyh_tracking.gif)

## Environment setup
```
conda env create -f environment.yml
```

Install the modified version of facexlib by
```
cd facexlib
pip install -e .
```

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
Put the video and csv_file (generated from ego4d_face_tracking.py) together (Within same folder).
then run:
```
python meta_tracking.py --old_csv_video_path (csv_file and video path) --filename (The video you are going to process)
```

