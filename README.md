# Text-to-Speech

### Setup

Choose a PyTorch container from [NVIDIA PyTorch Container Versions](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-22-11.html#rel-22-11) and create a Dockerfile as `/text2speech/docker/Dockerfile` 

- Build and run docker
```
$ docker build --no-cache -t torcht2s .
$ docker run -it --rm --gpus all -p 2222:8888 -v /your/working/directory/text-to-speech/text2speech:/your/working/directory/text-to-speech/text2speech torcht2s
```
- Add environment to jupyter notebook and launch jupyter notebook 
```
$ python -m ipykernel install --user --name=torcht2s
$ jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```
Open a browser from your local machine and navigate to `127.0.0.1:2222/?token=${TOKEN}` and enter your token specified in your terminal.

### Data Preperation

- Crate your data file as the following format and mount under the `text2speech/Fastpitch/dataset/` location.  
```
wav/42089-yeraltindanNotlar-8.wav|çocuklar da sevdikleri bir şey bekledikleri kimselere böyle bakarlar
```
- Preprocess the data
Run the pre-processing script to calculate pitch and mels `text2speech/Fastpitch/data_preperation.ipynb`
```
python prepare_dataset.py --wav-text-filelists dataset/tts_data.txt --n-workers 0 --batch-size 1 --dataset-path dataset --extract-pitch --f0-method pyin --extract-mels
```
The complete dataset has the following structure:
```
./dataset
|-mels
|-pitch
|-wavs
|-tts_data.txt
```










