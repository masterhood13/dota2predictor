FROM python:3.9-slim

LABEL maintainer="masterhood13 <masterhood13@gmail.com>" \
      platform="Linux" \
      description="Telegram bot that predicts Dota 2 match outcomes using XGBoost" \
      application_name="Dota 2 Predictor" \
      documentation="https://github.com/masterhood13/dota2predictor" \
      source="https://github.com/masterhood13/dota2predictor" \
      issues="https://github.com/masterhood13/dota2predictor/issues" \
      license="MIT" \
      vendor="masterhood13"

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "start.py"]
