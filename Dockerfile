FROM python:3.11.5-slim

WORKDIR /yacy_cf_backend

COPY . .

# RUN apt-get update && apt-get install -y \
#     python3 \
#     python3-pip

RUN pip3 install -r requirements.txt

CMD ["python3", "main.py"]

EXPOSE 3001
EXPOSE 8190