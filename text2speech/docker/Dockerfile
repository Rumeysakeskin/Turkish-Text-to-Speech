FROM continuumio/anaconda
WORKDIR /Path/to/working/directory/text2speech/

RUN conda create -n text2speech python=3.7

COPY environment.yml .
RUN conda env create -f environment.yml

RUN echo "source activate text2speech" > ~/.bashrc

# Set PATH environment variables
ENV PATH /opt/conda/envs/text2speech:$PATH
