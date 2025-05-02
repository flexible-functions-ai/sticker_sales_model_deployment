# Sticker Sales Forecasting with Modal and BentoML

In January 2025, Kaggle ran a playground series [competition](https://www.kaggle.com/competitions/playground-series-s5e1/overview) to forecast multiple years' worth of sticker sales in different countries. For each id row, you have to predict the num_sold which represents the number of stickers sold for each type per store for each country.

This notebook demonstrates how to build and deploy a machine learning solution for forecasting sticker sales using [BentoML](https://www.bentoml.com/) for model management and [Modal](https://modal.com/) for model training, deployment, and serving to an endpoint.

I previously showed how to train, serve, and deploy a machine learning model to a live API endpoint with only bentoml and bentocloud without Modal [here](https://nbsanity.com/static/ac40bb062434c7446906d3eb8875e061/load.html) 