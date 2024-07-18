FROM python:3.11.5-slim

WORKDIR /yacy_cf_backend

COPY . .

RUN pip3 install -r requirements.txt

CMD ["python3", "main.py"]