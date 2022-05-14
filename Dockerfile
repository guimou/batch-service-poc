FROM registry.access.redhat.com/ubi8/python-39:1-51

WORKDIR /usr/src/app

COPY train.py requirements.txt ./

RUN pip install -r requirements.txt

CMD ["python", "-u", "train.py"]
