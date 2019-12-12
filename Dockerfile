FROM python:3.6.9-stretch

# --------------------------------------------------------------------------------------------
# Install Cytomine python client
RUN git clone https://github.com/cytomine-uliege/Cytomine-python-client.git && \
    cd /Cytomine-python-client && git checkout tags/v2.3.0.poc.1 && pip install . && \
    rm -r /Cytomine-python-client

# --------------------------------------------------------------------------------------------
# Install Neubias-W5-Utilities (annotation exporter, compute metrics, helpers,...)
# Metric for PixCla is pure python so don't need java, nor binaries
RUN apt-get update && apt-get install libgeos-dev -y && apt-get clean
RUN git clone https://github.com/Neubias-WG5/neubiaswg5-utilities.git && \
    cd /neubiaswg5-utilities/ && git checkout tags/v0.8.7 && pip install . && \
    rm -r /neubiaswg5-utilities

# --------------------------------------------------------------------------------------------
# Install pytorch
RUN pip install https://download.pytorch.org/whl/cpu/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
RUN pip install torchvision==0.2.1

# --------------------------------------------------------------------------------------------
# Install scripts and models
ADD CP58_dice_0.9373_loss_0.0265.pth /app/CP58_dice_0.9373_loss_0.0265.pth
ADD descriptor.json /app/descriptor.json
ADD unet /app/unet
ADD wrapper.py /app/wrapper.py

ENTRYPOINT ["python", "/app/wrapper.py"]
