# Text-to-Speech

### Setup

Choose a PyTorch container from [NVIDIA PyTorch Container Versions](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-22-11.html#rel-22-11) and create a Dockerfile as `/text2speech/docker/Dockerfile` 

1. Build and run docker
```
$ docker build --no-cache -t torcht2s .
$ docker run -it --rm --gpus all -p 2222:8888 -v /your/working/directory/text-to-speech/text2speech:/your/working/directory/text-to-speech/text2speech torcht2s
```
2. Add environment to jupyter notebook and launch jupyter notebook 
```
$ python -m ipykernel install --user --name=torcht2s
$ jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```
3. Open a browser from your local machine and navigate to `127.0.0.1:2222/?token=${TOKEN}` and enter your token specified in your terminal.

### Data Preperation
Follow these steps to use custom dataset.
1. Prepare a directory with .wav files, filelists (training/validation split of the data) with transcripts and paths to .wav files under the `text2speech/Fastpitch/dataset/` location. Those filelists should list a single utterance per line as: 
```
<audio file path>|<transcript>
```
- Preprocess the data
2. Run the pre-processing script to calculate pitch and mels with `text2speech/Fastpitch/data_preperation.ipynb`
```
python prepare_dataset.py \ 
    --wav-text-filelists dataset/tts_data_train.txt \ 
                         dataset/tts_data_val.txt \
    --n-workers 0 \
    --batch-size 1 \
    --dataset-path dataset \
    --extract-pitch \
    --f0-method pyin \
    --extract-mels \
```
3. Prepare file lists with paths to pre-calculated pitch running `create_picth_text_file()` from `text2speech/Fastpitch/data_preperation.ipynb` 
Those filelists should list a single utterance per line as: 
```
<audio file path>|<audio pitch .pt file path>|<transcript>
```
The complete dataset has the following structure:
```
./dataset
├── mels
├── pitch
├── wavs
├── tts_data_train.txt
├── tts_data_val.txt
├── tts_data_pitch_train.txt
├── tts_data_pitch_val.txt
```

### Train Fastpitch from scratch (Spectrogram Generator)
The training will produce a FastPitch model capable of generating mel-spectrograms from raw text. It will be serialized as a single `.pt` checkpoint file, along with a series of intermediate checkpoints.
```
python train.py --cuda --amp --p-arpabet 1.0 --dataset-path dataset \ 
                --output saved_fastpicth_models/ \
                --training-files dataset/tts_data_pitch_train.txt \ 
                --validation-files dataset/tts_data_pitch_val.txt \ 
                --epochs 1000 --learning-rate 0.001 --batch-size 32 \
                --load-pitch-from-disk
```

### Fine-tune the model
Some mel-spectrogram generators are prone to model bias. As the spectrograms differ from the true data on which HiFi-GAN was trained, the quality of the generated audio might suffer. In order to overcome this problem, a HiFi-GAN model can be fine-tuned on the outputs of a particular mel-spectrogram generator in order to adapt to this bias. In this section we will perform fine-tuning to [FastPitch outputs](https://github.com/Rumeysakeskin/text2speech/blob/main/Fastpitch/saved_fastpitch_models/FastPitch_checkpoint.pt)

1. Generate mel-spectrograms for all utterances in the dataset with the FastPitch model
```
python extract_mels.py --cuda -o data/mels-fastpitch-tr22khz \ 
    --dataset-path /text2speech/Fastpitch/dataset \
    --dataset-files data/tts_pitch_data.txt --load-pitch-from-disk \
    --checkpoint-path data/pretrained_fastpicth_model/FastPitch_checkpoint.pt -bs 16
 ```
Mel-spectrograms should now be prepared in the data/mels-fastpitch-tr22khz directory. The fine-tuning script will load an existing HiFi-GAN model and run several epochs of training using spectrograms generated in the last step.

2. Fine-tune the Fastpitch model with HiFi-GAN 
This step will produce another .pt HiFi-GAN model checkpoint file fine-tuned to the particular FastPitch model.







