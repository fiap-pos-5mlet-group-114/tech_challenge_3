# Temperature Predictor

## Project Objective

The main objective of the project is to be a POC(prove of concept) on how using as minimum parameters as possible(latitude, longitude, altitude, hour, month, day) can still be viable on predicting(with regression) temperature on a certain place in the world.

## Project Def

The project was build with the following main libs

1. [FastAPI](https://fastapi.tiangolo.com/) for the api part
2. [PyTorch](https://pytorch.org/) for the model part
3. [Polars](https://pola.rs/) to deal with DataFrames
4. [SQLAlchemy](https://www.sqlalchemy.org/) for the database orm; with sqlite being the database

## The Data

All the data used to train the model was collected from the [National Institute of Meteorology of Brazil](https://portal.inmet.gov.br/dadoshistoricos), so there will be an bias towards brazilian climate(further testing shown that it still can predict quite well the temperature of other countries).

## Installation and Execution

First of all, to better better control the env and required libraries I used [uv](https://docs.astral.sh/uv/), so all the following steps require it installed.

After cloning the repo, run the following in your preferred terminal:

1. `uv sync --no-dev --extra gpu` to create the environment with all non dev required libs (remove the `--no-dev` to be able to run the non api scripts for both downloading the pre-trained model and the scripts for downloading and loading the dataset into the sql); change the `gpu` to `cpu` if your machine does not have a gpu accelerator(e.g. nvidia graphics card)
2. `uv run uvicorn src.server.config:app` to run the project api

## Development Process

While developing this project I've got into some problems both related to the data, and to the regression model itself;

### Talking about the data

The fist thing was the the data sources file were not even and csv(even though it's extracted from their zip file as that), the types were either wrong or badly formatted(in case of floats), so one of the things that took way much time than I was expecting was fixing these. The zip contains data from months and states with each line being an hour, so, it's needed to merge all of them into a single file to better use it later.

While only having the csv would be enough I wanted to make possible to use an API to create datasets and train the model with different datasets, so I build an script to populate an sql database with the data from the said csv containing all the data from the months and states.

### Talking about the model

If you give a quick view on the data you can see that there's way more fields that we could use in the project as input to predict the temperature, or as targets to predict. I didn't used them because i wanted to focus only on minimal data that a person can gather without the need of complex sensors, so, all the input can be simple gathered with an GPS, a clock and a calendar. As for the target, I wanted to simplify it as much as possible by predicting only one variable, that being the mean of the min and max temperature(found in the csv in different columns).

The model chosen was a simple MLP with the following structure, mainly because it was easier to implement. The `Linear` being the regression function(`f(x): (x * w) + b`) and the `ReLu` the activation function(`f(x): x if x > 0 else 0`).

```txt
Linear(6, 64)
ReLU()
Linear(64, 64)
ReLU()
Linear(64, 1)
```
