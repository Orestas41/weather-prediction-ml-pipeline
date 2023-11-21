FROM python:3.11.5

WORKDIR /weather_prediction_ml_pipeline

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN pip install wandb
ENV WANDB_API_KEY=[Your W&B API key]
RUN wandb login

EXPOSE 8080

CMD ["mlflow", "run", "."]
