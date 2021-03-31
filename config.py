"""Flask app configuration."""

class Config:
    """Base config."""
    SECRET_KEY = '1111abc123000abc1230001111'
    STATIC_FOLDER = 'static'
    TEMPLATES_FOLDER = 'templates'

class ProdConfig(Config):
    """ Production environment """
    FLASK_ENV = 'production'
    DEBUG = False
    TESTING = False

class DevConfig(Config):
    """ Development environment """
    FLASK_ENV = 'development'
    DEBUG = True
    TESTING = True