from flask import Flask
import pandas as pd

def create_app():
    """ 
    Initialize Flask application.
    """
    app = Flask(__name__,
                instance_relative_config=False)
    app.config.from_object('config.DevConfig')
    with app.app_context():
        from . import routes
        print(app.static_folder)
        return app