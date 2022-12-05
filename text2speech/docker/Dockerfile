FROM continuumio/anaconda
WORKDIR /Path/to/working/directory/text2speech/

COPY environment.yml .
RUN conda env create -f environment.yml

# Set PATH environment variables
ENV PATH /opt/conda/envs/text2speech:$PATH
