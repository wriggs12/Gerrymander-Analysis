FROM python

WORKDIR /workdir
COPY . .
RUN apt-get update && apt-get -y install cmake
RUN pip install -r ./requirements.txt

CMD ["python3", "./main.py", "nv", 543, 10]