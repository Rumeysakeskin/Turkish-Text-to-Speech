# Turkish Text-to-Speech
## Table Of Contents
- Setup
- Phonetical Conversion and Normalization for Turkish
- Data Preperation
- Training Fastpitch from scratch (Spectrogram Generator)
- Fine-tuning the model with HiFi-GAN (Waveforms Generator from input mel-spectrograms)

### Setup

Choose a PyTorch container from [NVIDIA PyTorch Container Versions](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-22-11.html#rel-22-11) and create a Dockerfile as following:
```
FROM nvcr.io/nvidia/pytorch:21.02-py3
WORKDIR /path/to/working/directory/text2speech/
COPY requirements.txt .
RUN pip install -r requirements.txt
```
1. Build and run docker
```
$ docker build --no-cache -t torcht2s .
$ docker run -it --rm --gpus all -p 2222:8888 -v /path/to/working/directory/text2speech:/path/to/working/directory/text2speech torcht2s
```
2. Add environment to jupyter notebook and launch jupyter notebook 
```
$ python -m ipykernel install --user --name=torcht2s
$ jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```
3. Open a browser from your local machine and navigate to `127.0.0.1:2222/?token=${TOKEN}` and enter your token specified in your terminal.

### Phonetical Conversion and Normalization for Turkish
In order to train speech synthesis models, sounds and phoneme sequences expressing sounds are needed. 
Since Turkish is a phonetic language, words are expressed as they are read. That is, character sequences are constructed words in Turkish. 
In non-phonetic languages such as English, words can be expressed with phonemes.
To synthesize Turkish speech with English data, the words in the English dataset first must be phonetically translated into Turkish. 
- In this study, [cmudict_tr](https://github.com/Rumeysakeskin/text2speech/blob/main/Fastpitch/cmudict/cmudict_tr) and [heteronyms_tr](https://github.com/Rumeysakeskin/text2speech/blob/main/Fastpitch/cmudict/heteronyms_tr) were used. CMUDict ([Turkish phonetic lexicon](https://github.com/DuyguA/computational_linguistics)) is a dictionary that phonetically expresses about 1.5M words in Turkish.
- The following phonemes represent the Turkish pronunciation of the phonemes.
```
valid_symbols = ['1', '1:', '2', '2:', '5', 'a', 'a:', 'b', 'c', 'd', 'dZ', 'e', 'e:', 'f', 'g', 'gj', 'h', 'i', 'i:', 'j',
  'k', 'l', 'm', 'n', 'N', 'o', 'o:', 'p', 'r', 's', 'S', 't', 'tS', 'u', 'u', 'v', 'y', 'y:', 'z', 'Z']
```
- Text normalization converts text from written form into its verbalized form, and it is an essential preprocessing step before text-to-speech synthesis.
It ensures that TTS can handle all input texts without skipping unknown symbols.
In this study, [text normalized](https://github.com/Rumeysakeskin/text2speech/blob/main/Fastpitch/common/text/turkish_text_normalization/turkish_text_normalizer.py) for Turkish utterances.


### Data Preperation
Follow these steps to use custom dataset.
1. Prepare a directory with .wav files, filelists (training/validation split of the data) with transcripts and paths to .wav files under the `text2speech/Fastpitch/dataset/` location. Those filelists should list a single utterance per line as: 
```
<audio file path>|<transcript>
```
- Preprocess the data
To speed-up training, those could be generated during the pre-processing step and read directly from the disk during training.

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
<mel or wav file path>|<pitch file path>|<text>|<speaker_id>
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

### Training Fastpitch from scratch (Spectrogram Generator)
The training will produce a FastPitch model capable of generating mel-spectrograms from raw text. It will be serialized as a single `.pt` checkpoint file, along with a series of intermediate checkpoints.
```
python train.py --cuda --amp --p-arpabet 1.0 --dataset-path dataset \ 
                --output saved_fastpicth_models/ \
                --training-files dataset/tts_data_pitch_train.txt \ 
                --validation-files dataset/tts_data_pitch_val.txt \ 
                --epochs 1000 --learning-rate 0.001 --batch-size 32 \
                --load-pitch-from-disk
```

### Fine-tuning the model with HiFi-GAN
Some mel-spectrogram generators are prone to model bias. As the spectrograms differ from the true data on which HiFi-GAN was trained, the quality of the generated audio might suffer. In order to overcome this problem, a HiFi-GAN model can be fine-tuned on the outputs of a particular mel-spectrogram generator in order to adapt to this bias. In this section we will perform fine-tuning to [FastPitch outputs](https://github.com/Rumeysakeskin/text2speech/blob/main/Fastpitch/saved_fastpitch_models/FastPitch_checkpoint.pt)

1. Generate mel-spectrograms for all utterances in the dataset with the FastPitch model
```
python extract_mels.py --cuda -o data/mels-fastpitch-tr22khz \ 
    --dataset-path /text2speech/Fastpitch/dataset \
    --dataset-files data/tts_pitch_data.txt --load-pitch-from-disk \
    --checkpoint-path data/pretrained_fastpicth_model/FastPitch_checkpoint.pt -bs 16
 ```
Mel-spectrograms should now be prepared in the `Hifigan/data/mels-fastpitch-tr22khz` directory. The fine-tuning script will load an existing HiFi-GAN model and run several epochs of training using spectrograms generated in the last step.

2. Fine-tune the Fastpitch model with HiFi-GAN 
This step will produce another `.pt` HiFi-GAN model checkpoint file fine-tuned to the particular FastPitch model.







