FROM docker.nju.edu.cn/pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime as builder

WORKDIR /app
COPY requirements-dev.txt /app/

RUN pip install -r requirements-dev.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN apt update
RUN apt install -y net-tools


FROM builder

COPY . /app

RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

EXPOSE 8000

CMD ["python", "-m", "src.api"]
