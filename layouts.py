import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import base64

colors = {"background": "#000000", "text": "#ffffff"}

test_png = './assets/hist_classifiers.png'
test_base64 = base64.b64encode(open(test_png, 'rb').read()).decode('ascii')

def layout():
    return html.Div(
        [
            html.Div([html.Center([html.H1("Deep unsupervised learning methods for TCR specificity"),
            html.Div(html.H5("Yanis Miraoui"),style={"color":"blue"}),
            html.Div(html.H5("CID: 01731821 / Imperial College London"),style={"color":"blue"}),
            html.Div(html.H5("yanis.miraoui19@imperial.ac.uk"),style={"color":"blue"}),
            ])]),
            html.Div([dbc.Tabs([dbc.Tab(home_tab(),label="Home")])]
        )])

def home_tab():
    return html.Div([html.Div([
                html.Div([
                            html.Div(
                                [dcc.Store(data=[], id="results_data"), dcc.Store(data=[], id="model_stats")]
                            ),
                            html.Div(
                                [
                                    html.I("Choose the representation model below:\n"),
                                    dcc.Dropdown(
                                        placeholder="Deep learning models",
                                        id="model_choices",
                                        multi=False,
                                    )
                                ],className="spaced_div"
                            ),
                            html.Div(
                                [
                                    html.Br(),
                                    html.Br(),
                                    html.I("Type your personalized CDR3 sequence below :\n"),
                                    html.Div([
                                    dcc.Input(
                                        placeholder="Type your CDR3 sequence here",
                                        type="text",
                                        value="",
                                        id="perso_cdr3sequence",
                                        style={'width': '90%'}
                                    )],),
                                    html.Div(id="output")
                                ],className="spaced_div"
                            ),
                            html.Div(
                                [   
                                    html.Br(),
                                    html.Br(),
                                    html.Div(html.I("Click this button if you want to use an existing CDR3 sequence at random :\n")),
                                    html.Div([
                                    dcc.Input(
                                        placeholder="Random CDR3 sequence will appear here",
                                        type="text",
                                        value="",
                                        id="random_cdr3sequence",
                                        style={'width': '90%'}
                                    )],),
                                    html.Div(html.Button('Generate CDR3 sequence', id='submit_val_random', n_clicks=0)),
                                    html.Div(id="output_random")
                                ],className="spaced_div"
                            ),
                        ],
                        className="pretty_container",
                    ),
                html.Div([
                            html.Div(
                                [
                                     html.Div([ html.Div(html.H3("Predicted cluster specificity:"),style={"font-size":"5.0rem"}), 
                                                html.Div(html.Center([],id="result_text",style={"font-size":"5.0rem"})), 
                                                html.Div(html.Center([],id="perform_stats",style={"font-size":"3.0rem"})),
                                                html.Br(),
                                                ]),
                                ],className="spaced_div pretty_container" 
                            ),
                        ],
                        className="four columns",
                    ),
                html.Div([
                            html.Div(
                                [
                                    html.Div(html.H3("Guidelines: "),style={"font-size":"5.0rem"}),
                                    html.Div(html.H5("1. Choose a model from which the latent space representation will be computed.")),
                                    html.Div(html.H5("2. Type your own CDR3 sequence or click on the 'Generate' button to generate an existing CDR3 sequence at random.")),
                                    html.Div(html.H5("3. Choose the v-gene and j-gene of your TCR from the large list provided.")),
                                    html.Div(html.H5("4. Click on the 'Predict' button to get the predicted representation of your TCR along with the results of its clustering.")),
                                    html.Div(html.H3("Results:"), style={"font-size":"5.0rem"}),
                                    html.Div(html.H5(" - The prediction embedding is computed and its reducted representation is shown along with random other sequences (using UMAP).")),
                                    html.Div(html.H5(" - The group to which the TCR is mostly to belong to is displayed along with the most represented antigen of that group.")),
                                    html.Div(html.H5(" - The clusters are determined using UMAP and K-Means clustering (with cross-validation for the choice of k).")),
                                    html.Div(html.H5("PLEASE NOTE: the website can sometimes be slow to load the text and the results. Please wait a few seconds for the content to load."),style={"color":"red"}),
                                ],className="spaced_div pretty_container" 
                            ),
                            html.Div(
                                [
                                    html.Div(html.H4("General performance of the models: "),style={"font-size":"3.0rem"}),
                                    html.Div([html.Img(src='data:image/png;base64,{}'.format(test_base64))]),
                                ],className="spaced_div pretty_container" 
                            )
                        ],
                        className="five columns",
                    ),
                ],style={"display":"inline-flex"}
                ),
            html.Div(
                [html.Div(html.Img(id="plot",style={"width":"80%", }))],
                className="spaced div pretty_container", style={'textAlign': 'center'}
            ),  
                ],style={"display":"inline-block"},id="main_frame_div"
            )

