from flask import Flask

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