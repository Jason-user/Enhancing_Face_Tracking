# Meta-Face-Tracking

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
Multi GPU (You can refer to Accelerate)
Setup .[Accelerate]_(https://huggingface.co/docs/accelerate/index).
```
accelerate config
accelerate launch ego4d_face_tracking.py --input_folder (your_video_path) --save_folder (output_dir)
```

