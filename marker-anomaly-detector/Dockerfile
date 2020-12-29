FROM continuumio/miniconda3

SHELL ["/bin/bash", "-c"]

RUN mkdir -p /PAD
ADD . /PAD
VOLUME ["/PAD"]
WORKDIR /PAD

RUN chmod +x set_uid.sh

ADD environment.yml /tmp/environment.yml
RUN chmod g+w /etc/passwd
RUN conda env create -f /tmp/environment.yml
ENV PATH /opt/conda/envs/mad-env/bin:$PATH

RUN /bin/bash -c "source activate mad-env"


USER 1001

CMD /PAD/set_uid.sh
