from flask import Flask
import pandas as pd

def create_app():
    """ 
    Initialize Flask application.
    """
    app = Flask(__name__,
                instance_relative_config=False)
    app.config.from_object('config.ProdConfig')
    with app.app_context():
        from . import routes
        return app