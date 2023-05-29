import dash
from dash.dependencies import Input, Output, State
import pandas as pd
from utils.helpers import *
import time
import tensorflow as tf
from dash import html


def callbacks(app):

    @app.callback(
    [   
        Output("output", "children"),
        Output("perso_cdr3sequence", "value"),
        Output("output_random", "children"),
        Output("random_cdr3sequence", "value"),
        Output("output_gene", "children"),
        Output("v_gene", "value"),
        Output("j_gene", "value")
    ],
    Input("perso_cdr3sequence", "value"),
    Input("results_data", "data"),
    Input("submit_val_random", "n_clicks"),
    Input("v_gene", "value"),
    Input("j_gene", "value")
    )
    def update_output(input, data, button1, v_gene, j_gene):
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if "submit_val_random" in changed_id:
            input = ""
            df = pd.DataFrame(data)
            row = df.sample(n=1)
            text = str(row["CDR3"].item())
            v_gene = str(row["v_gene"].item())
            j_gene = str(row["j_gene"].item())
            return [u'Personalized CDR3 sequence:\n {}'.format(input), input, u'Random CDR3 sequence:\n {}'.format(text), text, u'v-gene:\n {} \nj-gene:\n {}'.format(v_gene, j_gene), v_gene, j_gene]
        else :
            return [u'Personalized CDR3 sequence:\n {}'.format(input), input, "Random CDR3 sequence:\n", "", u'V-gene:\n {} \nJ-gene:\n {}'.format(v_gene, j_gene), v_gene, j_gene]
    
    @app.callback(
        [
            Output("results_data", "data"),
            Output("model_choices", "options"),
            Output("v_gene", "options"),
            Output("j_gene", "options"),
        ],
        Input("main_frame_div", "id"),
    )
    def load_mainframe(id):
        data = load_data()
        model_options = ["Simple AutoEncoder", "Optimized Deep AutoEncoder", "Variational AutoEncoder", "Transformers"]
        model_options = [{"label": val, "value": val} for val in model_options]

        v_gene = data["v_gene"].to_list()
        v_gene = [{"label": val, "value": val} for val in v_gene]

        j_gene = data["j_gene"].to_list()
        j_gene =[{"label": val, "value": val} for val in j_gene]

        return [data.to_dict("records"),
                model_options,
                v_gene, 
                j_gene]

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
            Input("v_gene", "value"),
            Input("j_gene", "value"),
        ],
    )
    def display_results_summary(perso_cdr3sequence, random_cdr3sequence, data, model_name, v_gene, j_gene):
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
                [f"Silhouette score: {perform_stats[0]:.2f}", html.Br(), f"Calinski-Harabasz score: {perform_stats[1]:.2f}", html.Br(),  f"Davies-Bouldin score: {perform_stats[2]:.2f}"]
        ]
