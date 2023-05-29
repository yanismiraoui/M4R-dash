import dash
from dash.dependencies import Input, Output, State
import pandas as pd
from utils.helpers import *
import time
import tensorflow as tf


def callbacks(app):

    @app.callback(
    [   
        Output("output", "children"),
        Output("perso_cdr3sequence", "value"),
        Output("output_random", "children"),
        Output("random_cdr3sequence", "value")
    ],
    Input("perso_cdr3sequence", "value"),
    Input("results_data", "data"),
    Input("submit_val_random", "n_clicks"),
    )
    def update_output(input, data, button1):
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if "submit_val_random" in changed_id:
            input = ""
            df = pd.DataFrame(data)
            row = df.sample(n=1)
            text = str(row["CDR3"].item())
            return [u'Personalized CDR3 sequence:\n {}'.format(input), input, u'Random CDR3 sequence:\n {}'.format(text), text]
        else :
            return [u'Personalized CDR3 sequence:\n {}'.format(input), input, "Random CDR3 sequence:\n", ""]
    
    @app.callback(
        [
            Output("results_data", "data"),
            Output("model_choices", "options"),
        ],
        Input("main_frame_div", "id"),
    )
    def load_mainframe(id):
        data = load_data()
        model_options = ["Simple AutoEncoder", "Optimized Deep AutoEncoder", "Variational AutoEncoder", "Transformers"]
        model_options = [{"label": val, "value": val} for val in model_options]

        return [data.to_dict("records"),
                model_options]

    @app.callback(
        [
            Output("plot", "src"),
            Output("result_text","children"),
            Output("perform_stats","children"),
        ],
        [
            Input("perso_cdr3sequence", "value"),
            Input("random_cdr3sequence", "value"),
            Input("results_data", "data"),
            Input("model_choices", "value"),
            
        ],
    )
    def display_results_summary(perso_cdr3sequence, random_cdr3sequence, data, model_name):
        audio = ""
        src = ""
        if not data:
            return dash.no_update, dash.no_update, dash.no_update
        data = pd.DataFrame(data)
        if model_name:
            models_dict = {"Simple AutoEncoder": "simple_ae", "Optimized Deep AutoEncoder": "deep_ae", "Variational AutoEncoder": "vae", "Transformers": "transformers"}
            model = tf.keras.models.load_model(f'./models/{models_dict[model_name]}.h5')
            encoder = tf.keras.models.load_model(f'./models/{models_dict[model_name]}_encoder.h5')
            test_png = f'./assets/performance_{model_name}.png'
        else:
            return ["", "No model specified", ""]
        if len(perso_cdr3sequence) > 10:
            time.sleep(3)
            if len(perso_cdr3sequence) > 20:
                return ["", "Sequence must have a length less than 20", ""]
            elif len(perso_cdr3sequence) < 20:
                perso_cdr3sequence = align_seqs(perso_cdr3sequence)
            X_test, sample = preprocess(data, perso_cdr3sequence, model_name)
            prediction = encoder.predict(X_test)
            img, target, target_perc, perform_stats = plot_clusters(sample, prediction, model_name)
        elif len(random_cdr3sequence) > 10:
            X_test, sample = preprocess(data, random_cdr3sequence, model_name)
            prediction = encoder.predict(X_test)
            img, target, target_perc, perform_stats = plot_clusters(sample, prediction, model_name)
        else:
            return ["", "No CDR3 sequence specified", ""]



        return [
                img,
                f" {target}, with {target_perc:.2f}% of the cluster having the same specificity",
                f"Silhouette score: {perform_stats[0]:.2f} \nCalinski-Harabasz score: {perform_stats[1]:.2f} \nDavies-Bouldin score: {perform_stats[2]:.2f}"
        ]
