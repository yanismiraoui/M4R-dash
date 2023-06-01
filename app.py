import dash
import dash_bootstrap_components as dbc
from layouts import layout
from callbacks import callbacks

external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets,assets_folder="static")


server = app.server
app.config.suppress_callback_exceptions = True

app.layout = layout()
app.title = "Deep unsupervised learning methods for TCR specificity"

callbacks(app)

if __name__ == "__main__":
    app.run_server(host="127.0.0.1", port="8050", debug=True)
