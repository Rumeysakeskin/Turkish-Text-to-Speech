FROM continuumio/anaconda
WORKDIR /Path/to/working/directory/text2speech/

RUN conda create -n brain4xt2s python=3.7

COPY brain4xt2s.yml .
RUN conda env create -f brain4xt2s.yml

RUN echo "source activate brain4xt2s" > ~/.bashrc

# Set PATH environment variables
ENV PATH /opt/conda/envs/brain4xt2s:$PATH
