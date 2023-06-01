import dash
from dash.dependencies import Input, Output, State
import pandas as pd
from utils_helpers.helpers import *
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
        model_options = ["Simple AutoEncoder", "Optimized Deep AutoEncoder", "Variational AutoEncoder", "Transformers (TCR-BERT)"]
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
            Input("nb_points", "value"),
            Input("v_gene", "value"),
            Input("j_gene", "value"),
        ],
    )
    def display_results_summary(perso_cdr3sequence, random_cdr3sequence, data, model_name, nb_points, v_gene, j_gene):
        time.sleep(1)
        if not data:
            return dash.no_update, dash.no_update, dash.no_update
        data = pd.DataFrame(data)
        if nb_points < 40:
            return ["", "Minimum number of points is 40", ""]
        if nb_points > 1000:
            return ["", "Maximum number of points is 1000", ""]
        if model_name:
            models_dict = {"Simple AutoEncoder": "simple_ae", "Optimized Deep AutoEncoder": "deep_ae", "Variational AutoEncoder": "vae", "Transformers (TCR-BERT)": "transformers"}
            if model_name != "Transformers":
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
                if v_gene == None:
                    return ["", "No v-gene specified", ""]
                if j_gene == None:
                    return ["", "No j-gene specified", ""]
            if model_name == "Transformers (TCR-BERT)":
                import model_utils
                X_test, sample = preprocess(data, perso_cdr3sequence, v_gene, j_gene, model_name, N=int(nb_points))
                prediction = model_utils.get_transformer_embeddings(
                                                                        model_dir="wukevin/tcr-bert",
                                                                        seqs=X_test,
                                                                        layers=[-7],
                                                                        method="mean",
                                                                        device=3,
                                                                    )
            
            else:
                X_test, sample = preprocess(data, perso_cdr3sequence, v_gene, j_gene, model_name, N=int(nb_points))
                prediction = encoder.predict(X_test)
            img, target, target_perc, perform_stats = plot_clusters(sample, prediction, model_name)
        elif len(random_cdr3sequence) > 10:
            if model_name == "Transformers (TCR-BERT)":
                import model_utils
                X_test, sample = preprocess(data, random_cdr3sequence, v_gene, j_gene, model_name, N=int(nb_points))
                prediction = model_utils.get_transformer_embeddings(
                                                                        model_dir="wukevin/tcr-bert",
                                                                        seqs=X_test,
                                                                        layers=[-7],
                                                                        method="mean",
                                                                        device=3,
                                                                    )
            else:
                X_test, sample = preprocess(data, random_cdr3sequence, v_gene, j_gene, model_name, N=int(nb_points))
                prediction = encoder.predict(X_test)
            img, target, target_perc, perform_stats = plot_clusters(sample, prediction, model_name)
        else:
            return ["", "No CDR3 sequence specified", ""]



        return [
                img,
                f" {target}, with {target_perc:.2f}% of the cluster having the same specificity",
                [f"Silhouette score: {perform_stats[0]:.2f}", html.Br(), f"Calinski-Harabasz score: {perform_stats[1]:.2f}", html.Br(),  f"Davies-Bouldin score: {perform_stats[2]:.2f}"]
        ]
    
    @app.callback(
    [   
        Output("output_compare", "children"),
        Output("perso_cdr3sequence_compare", "value"),
        Output("output_random_compare", "children"),
        Output("random_cdr3sequence_compare", "value"),
        Output("output_gene_compare", "children"),
        Output("v_gene_compare", "value"),
        Output("j_gene_compare", "value")
    ],
    Input("perso_cdr3sequence_compare", "value"),
    Input("results_data_compare", "data"),
    Input("submit_val_random_compare", "n_clicks"),
    Input("v_gene_compare", "value"),
    Input("j_gene_compare", "value")
    )
    def update_output_compare(input, data, button1, v_gene, j_gene):
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
            Output("results_data_compare", "data"),
            Output("v_gene_compare", "options"),
            Output("j_gene_compare", "options"),
        ],
        Input("main_frame_div_compare", "id"),
    )
    def load_mainframe(id):
        data = load_data()

        v_gene = data["v_gene"].to_list()
        v_gene = [{"label": val, "value": val} for val in v_gene]

        j_gene = data["j_gene"].to_list()
        j_gene =[{"label": val, "value": val} for val in j_gene]

        return [data.to_dict("records"),
                v_gene, 
                j_gene]

    @app.callback(
        [
            Output("status_seq","children"),
            Output("plot1", "src"),
            Output("plot2","src"),
            Output("plot3","src"),
            Output("plot4","src"),
            Output("plot1_description","children"),
            Output("plot2_description","children"),
            Output("plot3_description","children"),
            Output("plot4_description","children"),
        ],
        [
            Input("perso_cdr3sequence_compare", "value"),
            Input("random_cdr3sequence_compare", "value"),
            Input("results_data_compare", "data"),
            Input("nb_points_compare", "value"),
            Input("v_gene_compare", "value"),
            Input("j_gene_compare", "value"),
        ],
    )
    def compare_results(perso_cdr3sequence, random_cdr3sequence, data, nb_points, v_gene, j_gene):
        time.sleep(1)
        if not data:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if "compare_button" in changed_id:
            data = pd.DataFrame(data)
            if nb_points < 40:
                return ["Minimum number of points is 40", "", "", "", "", "", "", "", ""]
            if nb_points > 1000:
                return ["Maximum number of points is 1000", "", "", "", "", "", "", "", ""]
            
            if len(perso_cdr3sequence) > 10:
                time.sleep(3)
                if len(perso_cdr3sequence) > 20:
                    return ["Sequence must have a length less than 20", "", "", "", "", "", "", "", ""]
                elif len(perso_cdr3sequence) < 20:
                    perso_cdr3sequence = align_seqs(perso_cdr3sequence)
                    if v_gene == None:
                        return ["No v-gene specified", "", "", "", "", "", "", "", ""]
                    if j_gene == None:
                        return ["No j-gene specified", "", "", "", "", "", "", "", ""]

                models_dict = {"Simple AutoEncoder": "simple_ae", "Optimized Deep AutoEncoder": "deep_ae", "Variational AutoEncoder": "vae", "Transformers (TCR-BERT)": "transformers"}
                plots = {}
                for model_name in models_dict.keys():    
                    model = tf.keras.models.load_model(f'./models/{models_dict[model_name]}.h5')
                    encoder = tf.keras.models.load_model(f'./models/{models_dict[model_name]}_encoder.h5')
                    test_png = f'./assets/performance_{model_name}.png'
                    X_test, sample = preprocess(data, perso_cdr3sequence, v_gene, j_gene, model_name, N=int(nb_points))
                    prediction = encoder.predict(X_test)
                    img, target, target_perc, perform_stats = plot_clusters(sample, prediction, model_name)
                    plots[model_name] = [img, target, target_perc, perform_stats]
            elif len(random_cdr3sequence) > 10:
                models_dict = {"Simple AutoEncoder": "simple_ae", "Optimized Deep AutoEncoder": "deep_ae", "Variational AutoEncoder": "vae", "Transformers (TCR-BERT)": "transformers"}
                plots = {}
                for model_name in models_dict.keys():    
                    model = tf.keras.models.load_model(f'./models/{models_dict[model_name]}.h5')
                    encoder = tf.keras.models.load_model(f'./models/{models_dict[model_name]}_encoder.h5')
                    test_png = f'./assets/performance_{model_name}.png'
                    X_test, sample = preprocess(data, random_cdr3sequence, v_gene, j_gene, model_name, N=int(nb_points))
                    prediction = encoder.predict(X_test)
                    img, target, target_perc, perform_stats = plot_clusters(sample, prediction, model_name)
                    plots[model_name] = [img, target, target_perc, perform_stats]

            else:
                return ["No CDR3 sequence specified", "", "", "", "", "", "", "", ""]



            return [
                    "",
                    plots['Simple AutoEncoder'][0],
                    plots['Optimized Deep AutoEncoder'][0],
                    plots['Variational AutoEncoder'][0],
                    plots['Transformers (TCR-BERT)'][0],
                    [f"Silhouette score: {plots['Simple AutoEncoder'][3][0]}", html.Br(), f"Calinski-Harabasz score: {plots['Simple AutoEncoder'][3][1]}", html.Br(),  f"Davies-Bouldin score: {plots['Simple AutoEncoder'][3][2]}"],
                    [f"Silhouette score: {plots['Optimized Deep AutoEncoder'][3][0]}", html.Br(), f"Calinski-Harabasz score: {plots['Optimized Deep AutoEncoder'][3][1]}", html.Br(),  f"Davies-Bouldin score: {plots['Optimized Deep AutoEncoder'][3][2]}"],
                    [f"Silhouette score: {plots['Variational AutoEncoder'][3][0]}", html.Br(), f"Calinski-Harabasz score: {plots['Variational AutoEncoder'][3][1]}", html.Br(),  f"Davies-Bouldin score: {plots['Variational AutoEncoder'][3][2]}"],
                    [f"Silhouette score: {plots['Transformers (TCR-BERT)'][3][0]}", html.Br(), f"Calinski-Harabasz score: {plots['Transformers (TCR-BERT)'][3][1]}", html.Br(),  f"Davies-Bouldin score: {plots['Transformers'][3][2]}"],
                    ]
        else:
            return ["", "", "", "", "", "", "", "", ""]
