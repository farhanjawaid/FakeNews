FROM python:3.7

RUN mkdir fast-api

COPY requirements.txt requirements.txt
COPY fastapi /fast-api/

RUN pip3 install --upgrade pip
RUN pip install -r requirements.txt
WORKDIR /fast-api/

EXPOSE 8080
CMD ["uvicorn", "routes.api:app", "--host", "0.0.0.0", "--port", "8080"]