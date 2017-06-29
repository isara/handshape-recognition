FROM bvlc/caffe:cpu

RUN apt-get update && apt-get install -y software-properties-common

RUN add-apt-repository ppa:mc3man/xerus-media

RUN apt-get update && apt-get install -y --no-install-recommends \
        python-tk \
        ffmpeg

RUN pip install boto
RUN pip install jupyter

CMD jupyter notebook --allow-root
#docker build -t rizkyario/caffe:handshape-recognition docker/