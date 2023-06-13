from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
import base64

colors = {"background": "#000000", "text": "#ffffff"}

test_png = './assets/performance_metrics.png'
test_base64 = base64.b64encode(open(test_png, 'rb').read()).decode('ascii')

def layout():
    return html.Div(
        [
            html.Div([html.Center([html.H1("Deep unsupervised learning methods for TCR specificity 🧬"),
            html.Div(html.H5("Yanis Miraoui"),style={"color":"blue"}),
            html.Div(html.H5("CID: 01731821 / Imperial College London"),style={"color":"blue"}),
            html.Div(html.H5("yanis.miraoui19@imperial.ac.uk"),style={"color":"blue"}),
            ])]),
            html.Div([dbc.Tabs([dbc.Tab(home_tab(),label="Home 🏠"), dbc.Tab(compare_models(),label="Compare models ⚖️"), dbc.Tab(about_tab(),label="About 📄"), dbc.Tab(chat_tab(),label="Chatbot 🤖")])])
        ])

def home_tab():
    return html.Div([html.Div([
                html.Div([
                            html.Div(
                                [dcc.Store(data=[], id="results_data"), dcc.Store(data=[], id="model_stats")]
                            ),
                            html.Div(
                                [
                                    html.I("Number of points:\n"),
                                    html.Div([
                                    dcc.Input(
                                        placeholder="Number of points for the plot",
                                        type="number",
                                        value=100,
                                        id="nb_points",
                                        style={'width': '40%'}
                                    )],),
                                ],className="spaced_div"
                            ),
                            html.Div(
                                [
                                    html.Br(),
                                    html.Br(),
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
                            html.Div(
                                [   
                                    html.Br(),
                                    html.Br(),
                                    html.Div(html.I("Choose V-gene from list:\n")),
                                    dcc.Dropdown(
                                        placeholder="v-gene",
                                        id="v_gene",
                                        multi=False,
                                    ),
                                    html.Br(),
                                    html.Div(html.I("Choose J-gene from list:\n")),
                                    dcc.Dropdown(
                                        placeholder="j-gene",
                                        id="j_gene",
                                        multi=False,
                                    ),
                                    html.Div(id="output_gene"),
                                    html.Br(),
                                    html.Div(html.H5("Note that choosing the v-gene and j-gene has no impact on the results for the Simple AutoEncoder model.",style={"color":"red"})),
                                ],className="spaced_div"
                            ),
                        ],
                        className="pretty_container",
                    ),
                html.Div([
                            html.Div(
                                [
                                     html.Div([ html.Div(html.H3("Predicted cluster specificity:"),style={"font-size":"5.0rem"}), 
                                                html.Div(html.Center([],id="result_text",style={"font-size":"3.0rem"})), 
                                                html.Div(html.Center([],id="perform_stats",style={"font-size":"2.0rem"})),
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
                                    html.Div(html.H5("2. Type your own CDR3 sequence and choose the V-gene and J-gene or click on the 'Generate' button to generate an existing CDR3 sequence at random.")),
                                    html.Div(html.H5("3. Wait a few seconds for the results to be computed and the plot to be shown below.")),
                                    html.Div(html.H3("Results:"), style={"font-size":"5.0rem"}),
                                    html.Div(html.H5(" - The prediction embedding is computed and its reducted representation is shown along with random other sequences (using UMAP).")),
                                    html.Div(html.H5(" - The group to which the TCR belongs to is displayed along with the most represented antigen of that group.")),
                                    html.Div(html.H5(" - The clusters are determined using UMAP and K-Means clustering (with cross-validation for the choice of k).")),
                                    html.Div(html.H5("PLEASE NOTE: the website can sometimes be slow to load the text and the results. Please wait a few seconds for the content to load."),style={"color":"red"}),
                                ],className="spaced_div pretty_container" 
                            ),
                            html.Div(
                                [
                                    html.Div(html.H4("General performance of the models: "),style={"font-size":"3.0rem"}),
                                    html.Div([html.Img(src='data:image/png;base64,{}'.format(test_base64),style={"width":"100%", })]),
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

def compare_models():
    return html.Div([
                 html.Div([
                            html.Div(
                                [dcc.Store(data=[], id="results_data_compare"), dcc.Store(data=[], id="model_stats_compare")]
                            ),
                            html.Div(
                                [
                                    html.I("Number of points:\n"),
                                    html.Div([
                                    dcc.Input(
                                        placeholder="Number of points for the plot",
                                        type="number",
                                        value=100,
                                        id="nb_points_compare",
                                        style={'width': '40%'}
                                    )],),
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
                                        id="perso_cdr3sequence_compare",
                                        style={'width': '90%'}
                                    )],),
                                    html.Div(id="output_compare")
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
                                        id="random_cdr3sequence_compare",
                                        style={'width': '90%'}
                                    )],),
                                    html.Div(html.Button('Generate CDR3 sequence', id='submit_val_random_compare', n_clicks=0)),
                                    html.Div(id="output_random_compare")
                                ],className="spaced_div"
                            ),
                            html.Div(
                                [   
                                    html.Br(),
                                    html.Br(),
                                    html.Div(html.I("Choose V-gene from list:\n")),
                                    dcc.Dropdown(
                                        placeholder="v-gene",
                                        id="v_gene_compare",
                                        multi=False,
                                    ),
                                    html.Br(),
                                    html.Div(html.I("Choose J-gene from list:\n")),
                                    dcc.Dropdown(
                                        placeholder="j-gene",
                                        id="j_gene_compare",
                                        multi=False,
                                    ),
                                    html.Div(id="output_gene_compare"),
                                    html.Br(),
                                    html.Div(html.H5("Note that choosing the v-gene and j-gene has no impact on the results for the Simple AutoEncoder amd Transformers model.",style={"color":"red"})),
                                ],className="spaced_div"
                            ),
                            html.Div(
                                [
                                    html.Br(),
                                    html.Div(html.Button('Compare different models', id='compare_button', n_clicks=0)),
                                    html.Div(id="status_seq")
                                ],className="spaced_div"
                            ),
                        ],
                        className="pretty_container",
                    ),
            html.Div(
                [html.Div(html.Img(id="plot1",style={"width":"80%", })),
                html.Div([html.Center([],id="plot1_description",style={"font-size":"2.0rem"})]),  
                ],className="spaced div pretty_container",
            ),
            html.Div(
                [html.Div(html.Img(id="plot2",style={"width":"80%", })),
                html.Div([html.Center([],id="plot2_description",style={"font-size":"2.0rem"})]),
                ],className="spaced div pretty_container",
            ),
            html.Div(
                [html.Div(html.Img(id="plot3",style={"width":"80%", })),
                html.Div([html.Center([],id="plot3_description",style={"font-size":"2.0rem"})]),
                ],className="spaced div pretty_container",
            ),
            html.Div(
                [html.Div(html.Img(id="plot4",style={"width":"80%", })),
                html.Div([html.Center([],id="plot4_description",style={"font-size":"2.0rem"})]),
                ],className="spaced div pretty_container",
            ),
                ],style={"display":"inline-block"},id="main_frame_div_compare"
            )

    

def about_tab():
    return html.Div(
                    [
                        html.Div(html.H3("About the project: "),style={"font-size":"2.0rem"}),
                        html.Div(html.H5("The project was supervised by Dr. Barbara Bravi at Imperial College London in 2022-2023.")),
                        html.Br(),
                        html.Div(html.H3("About the data: "),style={"font-size":"2.0rem"}),
                        html.Div(html.H5("The data used for this project was obtained from the article 'A large-scale database of t-cell receptor beta (tcr) sequences and binding associations from natural and synthetic exposure to sars-cov-2'. The data can be obtained online and is publicly available.")),
                        html.Br(),
                        html.Div(html.H3("About the models: "),style={"font-size":"2.0rem"}),
                        html.Div(html.H5("The models used for this project are the ones investigated in the analysis 'Deep unsupervised learning methods for the identification and characterization of TCR specificity to Sars-Cov-2'. The building and the analysis of these models are available on GitHub: https://github.com/yanismiraoui")),
                        html.Br(),
                        html.Div(html.H3("About the website: "),style={"font-size":"2.0rem"}),
                        html.Div(html.H5("The website was developed using Python, AWS, Heroku and Replit.")),
                    ],className="pretty_container", style={'textAlign': 'center'}
                ),


def chat_tab():
    return html.Div(
                    [
                        html.Div(html.H3("Research Paper Chatbot 🤖: "),style={"font-size":"2.0rem"}),
                        html.Iframe(src="https://www.chatbase.co/chatbot-iframe/4pvB3Q1vd6kdKNHxwy7-Z",
                        style={"height": "700px", "width": "100%"})
                    ],className="pretty_container", style={'textAlign': 'center'}
                )
