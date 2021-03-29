FROM jupyter/minimal-notebook:latest

# must reset to user root to install more stuff
USER root
# apt-utils is missing and needed to avoid warning about skipping debconf
RUN apt-get update && apt-get --yes install apt-utils

RUN apt install -y \
    libgl1-mesa-glx

# set the user back to original setting
USER $NB_UID

RUN pip install ipywidgets && jupyter nbextension enable --py --sys-prefix widgetsnbextension
