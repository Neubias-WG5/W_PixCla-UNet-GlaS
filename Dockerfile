FROM neubiaswg5/neubias-base:latest

RUN pip install https://download.pytorch.org/whl/cpu/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
RUN pip install torchvision
RUN pip install pydensecrf

ADD unet /app/unet
ADD wrapper.py /app/wrapper.py
ADD CP58_dice_0.9373_loss_0.0265.pth /app/CP58_dice_0.9373_loss_0.0265.pth
ADD descriptor.json /app/descriptor.json

ENTRYPOINT ["python", "/app/wrapper.py"]
