import gradio as gr
import os
import pandas as pd
import pickle as pkl


df = pd.read_csv('../data/filter-data-cleaned.csv')

try:
    model_f7 = pkl.load(open(os.path.join('..', 'models', 'best_model_f7.pkl'), 'rb'))
    model_m5 = pkl.load(open(os.path.join('..', 'models', 'best_model_m5.pkl'), 'rb'))
    model_g4 = pkl.load(open(os.path.join('..', 'models', 'best_model_g4.pkl'), 'rb'))
except FileNotFoundError:
    print('Please run the notebooks gradio_train_*.ipynb first to train the models.')


def get_similar_products(filter_efficiency, dim_length, dim_height, dim_gutter):
    default_error = 50
    filter_efficiency_mask = df['filter_efficiency'] == filter_efficiency
    length_mask = (df['Length'] > dim_length - default_error) & (df['Length'] < dim_length + default_error)
    height_mask = (df['Height'] > dim_height - default_error) & (df['Height'] < dim_height + default_error)
    gutter_mask = (df['Gutter'] > dim_gutter - default_error) & (df['Gutter'] < dim_gutter + default_error)

    return df[filter_efficiency_mask & length_mask & height_mask & gutter_mask][
        ['filter_efficiency', 'Length', 'Height', 'Gutter', 'unit_price']][:5]


def create_predictions_file():
    if not os.path.exists('predictions.txt'):
        with open('predictions.txt', 'w') as f:
            f.write('Filter Efficiency,Length,Height,Gutter,Predicted Price,Correct Price\n')


def save_prediction(filter_efficiency, dim_length, dim_height, dim_gutter, predicted_price, correct_price):
    create_predictions_file()

    with open('predictions.txt', 'a') as f:
        f.write(f'{filter_efficiency},{dim_length},{dim_height},{dim_gutter},{predicted_price:.2f},{correct_price:.2f}\n')


with gr.Blocks() as demo:
    gr.Markdown("## Make a prediction")

    with gr.Row():
        with gr.Column():
            filter_efficiency = gr.Dropdown(['F7', 'M5', 'G4'], value='F7', label="Filter Efficiency", interactive=True)

            dim_length = gr.Number(label="Length")
            dim_height = gr.Number(label="Height")
            dim_gutter = gr.Number(label="Gutter")

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
        gr.Markdown("## Similar products")
        similar_products = gr.Dataframe(interactive=False)

    def show_feedback_row():
        return {
            feedback_row: gr.Row(visible=True),
        }

    def show_correct_price_row():
        return {
            correct_price_row: gr.Row(visible=True),
        }

    def make_prediction(filter_efficiency, dim_length, dim_height, dim_gutter):
        if filter_efficiency == 'F7':
            price = model_f7.predict([[1, dim_length, dim_height, dim_gutter]])[0]
        elif filter_efficiency == 'M5':
            price = model_m5.predict([[1, dim_length, dim_height, dim_gutter]])[0]
        elif filter_efficiency == 'G4':
            price = model_g4.predict([[1, dim_length, dim_height, dim_gutter]])[0]
        else:
            raise Exception(f'Unknown filter efficiency: {filter_efficiency}')

        return {
            feedback_row: gr.Row(visible=True),
            feedback_label: gr.Label(f'Predicted price: {price:.2f}€'),
            correct_price_row: gr.Row(visible=False),
            predicted_price: gr.Number(visible=False, value=price),
            similar_products: gr.Dataframe(
                get_similar_products(filter_efficiency, dim_length, dim_height, dim_gutter),
            )
        }

    def submit_ok_prediction(filter_efficiency, dim_length, dim_height, dim_gutter, predicted_price):
        gr.Info("Feedback noted!")
        save_prediction(filter_efficiency, dim_length, dim_height, dim_gutter, predicted_price, predicted_price)

        return {
            feedback_row: gr.Row(visible=False),
            correct_price_row: gr.Row(visible=False),
        }

    def submit_correct_price(filter_efficiency, dim_length, dim_height, dim_gutter, predicted_price, correct_price):
        gr.Info("Feedback noted!")
        save_prediction(filter_efficiency, dim_length, dim_height, dim_gutter, predicted_price, correct_price)

        return {
            feedback_row: gr.Row(visible=False),
            correct_price_row: gr.Row(visible=False),
        }


    submit.click(make_prediction,
                 [filter_efficiency, dim_length, dim_height, dim_gutter],
                 [feedback_row, feedback_label, correct_price_row, predicted_price, similar_products])

    price_ok.click(submit_ok_prediction,
                   [filter_efficiency, dim_length, dim_height, dim_gutter, predicted_price],
                   [feedback_row, correct_price_row])

    price_not_ok.click(show_correct_price_row,
                       [], [correct_price_row])

    submit_correct_price_btn.click(submit_correct_price,
                                   [filter_efficiency, dim_length, dim_height, dim_gutter, predicted_price, correct_price],
                                   [feedback_row, correct_price_row])

demo.launch(server_port=1234)
