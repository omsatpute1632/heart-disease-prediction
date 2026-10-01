"""
CardioPulse AI — Root Application Runner
Run locally with: python app.py
"""

import uvicorn
from api.index import app

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🚀 CardioPulse Full-Stack Python App Starting...")
    print("  👉 Open your browser at: http://127.0.0.1:8000")
    print("  📊 API Docs available at: http://127.0.0.1:8000/docs")
    print("=" * 60 + "\n")
    uvicorn.run("api.index:app", host="127.0.0.1", port=8000, reload=True)
