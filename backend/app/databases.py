import os
from functools import lru_cache

# load_dotenv reads values from backend/.env during local development.
from dotenv import load_dotenv

# create_client creates the Supabase client used for database and storage calls.
from supabase import create_client

# Load environment variables before trying to read them.
load_dotenv()


# lru_cache keeps one Supabase client instance instead of creating a new one each call.
@lru_cache
def get_supabase():
    # SUPABASE_SERVICE_ROLE is preferred for backend writes.
    # SUPABASE_KEY is accepted as a fallback because your .env already has it.
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE") or os.getenv("SUPABASE_KEY")

    # Fail with a clear message if required Supabase credentials are missing.
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE or SUPABASE_KEY must be set before using Supabase."
        )

    # Return a connected Supabase client.
    return create_client(url, key)
