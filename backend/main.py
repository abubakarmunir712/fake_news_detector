import os
import sys

# Allow running from inside backend directory or root
# This adds the project root to sys.path so 'backend' package can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=app.config.get("PORT", 8000), debug=True)
