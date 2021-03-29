docker run \
  -d \
  -p 8888:8888 \
  -v /media/hdd/home/andrew/kaggle-data/shopee-product-matching:/data \
  -v /home/andrei/projects/shopee/notebooks:/home/jovyan/notebooks \
  --name=jupyter-shopee \
  --ipc=host \
  --gpus=all \
  jupyter-shopee-i
