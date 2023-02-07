FROM tensorflow/tensorflow

# RUN apt-get update
RUN mkdir ./api
WORKDIR /api
COPY . .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r requirements.txt
CMD ["uvicorn", "api:app","--host","0.0.0.0", "--port", "8000"]
