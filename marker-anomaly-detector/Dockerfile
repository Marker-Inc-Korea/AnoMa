FROM continuumio/miniconda3

SHELL ["/bin/bash", "-c"]

RUN mkdir -p /PAD
ADD . /PAD
VOLUME ["/PAD"]
WORKDIR /PAD

# RUN chmod +x /PAD

# SHELL ["/bin/bash", "-c"]

# ADD . /
RUN chmod +x set_uid.sh
# RUN chmod +x app.py

ADD environment.yml /tmp/environment.yml
RUN chmod g+w /etc/passwd
RUN conda env create -f /tmp/environment.yml
ENV PATH /opt/conda/envs/mad-env/bin:$PATH

# ENTRYPOINT ["/bin/sh", "-c", "/bin/bash"]
RUN /bin/bash -c "source activate mad-env"
# RUN "source activate prophet-env"

# RUN groupadd -g 999 1001
# RUN useradd -r -u 999 -g 1001 1001

USER 1001

# Ensure that assigned uid has entry in /etc/passwd.
# CMD ./set_uid.sh && /opt/conda/envs/prophet-env/bin/python ${APP_FILE}
# CMD /home/PAD/set_uid.sh && /opt/conda/envs/prophet-env/bin/python ${APP_FILE}
 


# CMD /PAD/set_uid.sh && python /PAD/${APP_FILE}
CMD /PAD/set_uid.sh


# CMD ["/PAD/set_uid.sh","/opt/conda/envs/prophet-env/bin/python && ${APP_FILE}"]


# FROM python:3.7
  
# RUN apt-get update
# RUN pip install mlflow
# RUN pip install mysqlclient

# RUN mkdir -p /home/mlflow
# COPY src/run_server.sh /home/mlflow
# WORKDIR /home/mlflow

# ARG ARG_MLFLOW_ARTIFACT_URI
# RUN mkdir -p $ARG_MLFLOW_ARTIFACT_URI

# ENTRYPOINT cd /home/mlflow ; ./run_server.sh 