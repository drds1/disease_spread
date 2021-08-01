FROM amancevice/pandas:0.23.4-python3-slim

WORKDIR /Main

COPY requirements.txt ./
COPY main.py ./
COPY disease_spread /Main/
COPY results /Main/

RUN pip install --no-cache-dir -r requirements.txt

#CMD ["python /main/main.py"]
