import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dash import Dash
import dash_bootstrap_components as dbc

from frontend import association_explorer_layout

# In the global app, this list will be determined dynamically from the query.
QUERY_VARIABLES = None  # None = toutes les variables du dataset

app = Dash(external_stylesheets=[dbc.themes.SANDSTONE])

association_explorer_layout.init(selected_variables=QUERY_VARIABLES)
app.layout = association_explorer_layout.create_layout()

if __name__ == "__main__":
    app.run(debug=True)