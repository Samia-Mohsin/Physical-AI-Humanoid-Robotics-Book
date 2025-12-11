# Vercel entry point for the unified server
import sys
import os

# Change to the backend directory to ensure imports work correctly
backend_dir = os.path.join(os.path.dirname(__file__), '..', 'backend')
sys.path.insert(0, backend_dir)
os.chdir(backend_dir)

# Import the app from the unified server
from unified_server_final import app

# This is the entry point for Vercel
app_instance = app