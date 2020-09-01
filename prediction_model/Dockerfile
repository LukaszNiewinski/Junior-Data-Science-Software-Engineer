FROM python:3.8.0-alpine

WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install dependencies
RUN pip install --upgrade pip
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# run entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]

CMD [ "python", "run.py" ]
