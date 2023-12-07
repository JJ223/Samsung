import gradio as gr
import os
import pandas as pd
import pickle as pkl
import json

from sklearn.metrics import pairwise_distances

import torch
import torch.nn as nn


manifest = json.load(open('models/gradio_manifest.json', 'r'))

best_model_f7 = manifest['best_model_f7']
best_model_m5 = manifest['best_model_m5']
best_model_g4 = manifest['best_model_g4']
best_model_general = manifest['best_model_general']


df = pd.read_csv(os.path.join('data', 'filter-data-cleaned.csv'))

features = ('qty', 'G4', 'M5', 'G3', 'F7', 'G2',
       'F9', 'M6', 'H14', 'F8', 'H13', 'Length', 'Height', 'Gutter')

filter_features = ('G4', 'M5', 'G3', 'F7', 'G2', 'F9', 'M6', 'H14', 'F8', 'H13')

available_filters = ('M5', 'M6', 'F7', 'F8', 'F9', 'G2', 'G3', 'G4', 'H13', 'H14')


def create_nn_model():
    return nn.Sequential(
        nn.Linear(14, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 1)
    )

try:
    if best_model_f7 == 'NeuralNetwork':
        model_f7 = create_nn_model()
        model_f7.load_state_dict(torch.load(os.path.join('models', 'best_model_f7.pt')))
    else:
        model_f7 = pkl.load(open(os.path.join('models', 'best_model_f7.pkl'), 'rb'))

    if best_model_m5 == 'NeuralNetwork':
        model_m5 = create_nn_model()
        model_m5.load_state_dict(torch.load(os.path.join('models', 'best_model_m5.pt')))
    else:
        model_m5 = pkl.load(open(os.path.join('models', 'best_model_m5.pkl'), 'rb'))

    if best_model_g4 == 'NeuralNetwork':
        model_g4 = create_nn_model()
        model_g4.load_state_dict(torch.load(os.path.join('models', 'best_model_g4.pt')))
    else:
        model_g4 = pkl.load(open(os.path.join('models', 'best_model_g4.pkl'), 'rb'))

    if best_model_general == 'NeuralNetwork':
        model_general = create_nn_model()
        model_general.load_state_dict(torch.load(os.path.join('models', 'best_model_general.pt')))
    else:
        model_general = pkl.load(open(os.path.join('models', 'best_model_general.pkl'), 'rb'))

except FileNotFoundError:
    print('Please make sure there are models in the models folder. You can train these with the Models_new.ipynb notebook.')


def get_similar_products(filter_efficiency, dim_length, dim_height, dim_gutter):

    df_search = df[df['filter_efficiency'] == filter_efficiency][['Length', 'Height', 'Gutter']]

    search_vector = [[dim_length, dim_height, dim_gutter]]

    # Apply cosine similarity
    similarity = pairwise_distances(df_search, search_vector)
    similarities = pd.Series(similarity.reshape(-1), index=df_search.index)

    # Get the top 5 most similar products
    return df.iloc[similarities.sort_values().index[:5]][['filter_efficiency', 'Length', 'Height', 'Gutter', 'qty', 'unit_price']]


def create_predictions_file():
    if not os.path.exists('predictions.txt'):
        with open('predictions.txt', 'w') as f:
            f.write('Filter Efficiency,Length,Height,Gutter,Quantity,Predicted Price,Correct Price\n')


def save_prediction(filter_efficiency, dim_length, dim_height, dim_gutter, quantity, predicted_price, correct_price):
    create_predictions_file()

    with open('predictions.txt', 'a') as f:
        f.write(f'{filter_efficiency},{dim_length},{dim_height},{dim_gutter},{quantity},{predicted_price:.2f},{correct_price:.2f}\n')


with gr.Blocks() as demo:
    gr.Markdown("## Make a prediction")

    with gr.Row():
        with gr.Column():
            filter_efficiency = gr.Dropdown(available_filters, value='F7', label="Filter Efficiency", interactive=True)

            dim_length = gr.Number(label="Length")
            dim_height = gr.Number(label="Height")
            dim_gutter = gr.Number(label="Gutter")
            quantity = gr.Number(label="Quantity", minimum=1, value=1)

            submit = gr.Button("Submit", variant="primary")

        with gr.Column():
            predicted_price = gr.Number(visible=False, value=0)
            feedback_label = gr.Label("Submit a prediction first", label="Predicted")

            with gr.Row(visible=False) as feedback_row:
                with gr.Column():
                    price_ok = gr.Button('Price is Correct', size='sm', variant='secondary')

                with gr.Column():
                    price_not_ok = gr.Button('Price is NOT Correct', size='sm', variant='stop')

            with gr.Row(visible=False) as correct_price_row:
                with gr.Column():
                    correct_price = gr.Number(label="Correct Price", minimum=0)
                    submit_correct_price_btn = gr.Button("Submit", variant="primary")

    with gr.Column():
        gr.Markdown("## Closest products")
        similar_products = gr.Dataframe(interactive=False)

    def show_feedback_row():
        return {
            feedback_row: gr.Row(visible=True),
        }

    def show_correct_price_row():
        return {
            correct_price_row: gr.Row(visible=True),
        }

    def make_prediction(filter_efficiency, dim_length, dim_height, dim_gutter, quantity):
        if filter_efficiency == 'F7':
            if manifest['best_model_f7'] == 'NeuralNetwork':
                price = model_f7(torch.tensor([[quantity, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, dim_length, dim_height, dim_gutter]]).float()).item()
            else:
                price = model_f7.predict([[quantity, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, dim_length, dim_height, dim_gutter]])[0]
        elif filter_efficiency == 'M5':
            if manifest['best_model_m5'] == 'NeuralNetwork':
                price = model_m5(torch.tensor([[quantity, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, dim_length, dim_height, dim_gutter]]).float()).item()
            else:
                price = model_m5.predict([[quantity, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, dim_length, dim_height, dim_gutter]])[0]
        elif filter_efficiency == 'G4':
            if manifest['best_model_g4'] == 'NeuralNetwork':
                price = model_g4(torch.tensor([[quantity, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, dim_length, dim_height, dim_gutter]]).float()).item()
            else:
                price = model_g4.predict([[quantity, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, dim_length, dim_height, dim_gutter]])[0]
        else:
            idx = filter_features.index(filter_efficiency)
            one_hot = [0] * len(available_filters)
            one_hot[idx] = 1

            if manifest['best_model_general'] == 'NeuralNetwork':
                price = model_general(torch.tensor([[quantity, *one_hot, dim_length, dim_height, dim_gutter]]).float()).item()
            else:
                price = model_general.predict([[quantity, *one_hot, dim_length, dim_height, dim_gutter]])[0]

        return {
            feedback_row: gr.Row(visible=True),
            feedback_label: gr.Label(f'Predicted price: {price:.2f}€'),
            correct_price_row: gr.Row(visible=False),
            predicted_price: gr.Number(visible=False, value=price),
            similar_products: gr.Dataframe(
                get_similar_products(filter_efficiency, dim_length, dim_height, dim_gutter),
            )
        }

    def submit_ok_prediction(filter_efficiency, dim_length, dim_height, dim_gutter, quantity, predicted_price):
        gr.Info("Feedback noted!")
        save_prediction(filter_efficiency, dim_length, dim_height, dim_gutter, quantity, predicted_price, predicted_price)

        return {
            feedback_row: gr.Row(visible=False),
            correct_price_row: gr.Row(visible=False),
        }

    def submit_correct_price(filter_efficiency, dim_length, dim_height, dim_gutter, quantity, predicted_price, correct_price):
        gr.Info("Feedback noted!")
        save_prediction(filter_efficiency, dim_length, dim_height, dim_gutter, quantity, predicted_price, correct_price)

        return {
            feedback_row: gr.Row(visible=False),
            correct_price_row: gr.Row(visible=False),
        }


    submit.click(make_prediction,
                 [filter_efficiency, dim_length, dim_height, dim_gutter, quantity],
                 [feedback_row, feedback_label, correct_price_row, predicted_price, similar_products])

    price_ok.click(submit_ok_prediction,
                   [filter_efficiency, dim_length, dim_height, dim_gutter, quantity, predicted_price],
                   [feedback_row, correct_price_row])

    price_not_ok.click(show_correct_price_row,
                       [], [correct_price_row])

    submit_correct_price_btn.click(submit_correct_price,
                                   [filter_efficiency, dim_length, dim_height, dim_gutter, quantity, predicted_price, correct_price],
                                   [feedback_row, correct_price_row])

demo.launch()
