# BeatDance

Code for ICMR 2024 paper "BeatDance: A Beat-Based Model-Agnostic Contrastive Learning Framework for Music-Dance Retrieval"

[[Paper]](TODO) | [[Video Demo]](https://youtu.be/xAIB5ucYiuI?si=EwNu8BKcz8Y_jELx)

<a href="https://www.youtube.com/watch?v=xAIB5ucYiuI" target="_blank">
    <img src="https://github.com/XulongT/BeatDance/blob/main/demo/playbutton.png" alt="Watch the video" width="400"/>
</a>



# Code

## Set up code environment

To set up the necessary environment for running the project, follow these steps:

1. **Create a new conda environment**:   

   ```
   conda create -n BD_env python=3.8
   conda activate BD_env
   ```

2. **Install PyTorch and dependencies**
   ```
   conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
   conda install --file requirements.txt
   ```

## Download

Directly download our data from [here](todo) as ./data folder.

To test with our pretrained models, please download the model from [here](https://drive.google.com/file/d/1bBp_PuK7_7y9VNYadW3w-HRRfW0wZmA2/view?usp=drive_link) (Google Drive)  into ./outputs folder.

## Directory Structure

After downloading the corresponding data and models, please move the relevant files to their respective directories.

The file directory structure is as follows:

```
|-- config
|-- data
    |--- dance_video
    |--- music_beat
    |--- music_feature
    |--- video_beat
    |--- video_feature
|-- datasets
|-- logs
|-- model
|-- modules
|-- outputs
    |--- BeatDance
|-- preprocess
|-- result
|-- trainer
```

## Training

```python
nohup python train.py --exp_name=BeatDance --videos_dir=./data/Music-Dance --batch_size=32  --beta 0.4 \
                --huggingface --dataset_name=MSRVTT --num_epochs 150 --num_frames 10 --dropout1 0.3 --dropout2 0.6 > ./result/train.log 2>&1 &
```

## Evaluation

In the testing code, we provide two testing scripts: '[test.py](https://github.com/XulongT/BeatDance/blob/main/test.py)' for non-QB-Norm versions, and '[test_qb_norm.py](https://github.com/XulongT/BeatDance/blob/main/test_qb_norm.py)' for QB-Norm versions. In 'test_qb_norm.py', you can adjust the 'mode' parameter to select the QB-Norm calculation mode. 

### 1. Common Mode

For Music-Video Retrieval:

    nohup python test.py --exp_name=BeatDance --videos_dir=./data/Music-Dance --batch_size=32  --beta 0.4 \
                    --load_epoch -1 --num_frames 10 > ./result/m2v_test.log 2>&1 &

For Video-Music Retrieval:

    nohup python test.py --exp_name=BeatDance --videos_dir=./data/Music-Dance --batch_size=32  --beta 0.4 \
                    --load_epoch -1 --metric v2t --num_frames 10 > ./result/v2m_test.log 2>&1 &

### 2. QB-Norm Mode

For Music-Video Retrieval:

    nohup python test_qb_norm.py --exp_name=BeatDance --videos_dir=./data/Music-Dance \
                    --beta 0.4 --load_epoch -1 --num_frames 10 --qbnorm_k 1 --qbnorm_beta 20 \
                    --qbnorm_mode train > ./result/m2v_test_qbnorm.log 2>&1 &

For Video-Music Retrieval:

```
nohup python test_qb_norm.py --exp_name=BeatDance --videos_dir=./data/Music-Dance \
                --beta 0.4 --load_epoch -1 --num_frames 10 --qbnorm_k 1 --qbnorm_beta 20 \
                --qbnorm_mode train --metric v2t > ./result/v2m_test_qbnorm.log 2>&1 &
```

If you have any question, don't hesitate to submit issue or contact me.

# Acknowledgments

Our code is based on [XPool](https://github.com/layer6ai-labs/xpool) . We sincerely appreciate for their contributions.

# Citation

    @inproceedings{todo
    }

