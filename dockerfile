FROM pytorchlightning/pytorch_lightning:base-cuda-py3.10-torch2.0-cuda11.7.1
COPY . /llm-test
WORKDIR /llm-test
RUN pip install -r requirements.txt
ENTRYPOINT [ "python","main.py" ]