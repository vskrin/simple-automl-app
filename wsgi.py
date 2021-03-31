"""
Run Flask application.
"""
import os
from app import create_app

app = create_app()
# pick up dynamic Heroku port if available, otherwise use port 5000 (local testing)
port = int(os.environ.get("PORT", 5000)) 

if __name__ == '__main__':
    # use host "0.0.0.0" for Heroku and "localhost" for local Docker testing
    app.run(host="0.0.0.0",
            port=port)