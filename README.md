# Medical Image Analysis Laboratory

Welcome to the medical image analysis laboratory (MIALab).
This repository contains all code you will need to get started with classical medical image analysis.

During the MIALab you will work on the task of brain tissue segmentation from magnetic resonance (MR) images.
We have set up an entire pipeline to solve this task, specifically:

- Pre-processing
- Registration
- Feature extraction
- Voxel-wise tissue classification
- Post-processing
- Evaluation

After you complete the exercises, dive into the 
    
    pipeline.py 

script to learn how all of these steps work together. 

During the laboratory you will get to know the entire pipeline and investigate one of these pipeline elements in-depth.
You will get to know and to use various libraries and software tools needed in the daily life as biomedical engineer or researcher in the medical image analysis domain.

Enjoy!

----

Found a bug or do you have suggestions? Open an issue or better submit a pull request.

# 102475-HS2024-0: Medical Image Analysis Lab

This is the repository for our MIA Lab pipeline-project submission.

### Group Members
José Miguel PINTO INÁCIO | jose.inacio@students.unibe.ch

Max BÖGLI | max.boegli@students.unibe.ch

Marco Daniel PORTMANN | marco.portmann@students.unibe.ch

## Executing the pipeline

The dataset (atlas, train and test) is in the folder "dataset". To execute the pipeline we must add parser arguments to this folder:

NOTE: Make sure to call this while being in the mialab folder, the paths are relative.

NOTE: In the future we might want to add the aboslute path as default parser arguments.

```bash
python pipeline.py --data_atlas_dir "./dataset/atlas" --data_train_dir "./dataset/train" --data_test_dir "./dataset/test"
```
After execution the pipeline script will create an output folder called "mia-results".

```bash
python pipeline.py --postprocess "inferences/2024-11-10-23-09-30.joblib"
```

NOTE: This folder was added to the .gitignore.


## Useful links

-   [MIA24 Slack](mialab2024.slack.com)


January 2025
