"""FastAPI dependencies: Supabase client, auth."""
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordBearer
from supabase import create_client, Client

from api.config import settings

security = HTTPBearer(auto_error=False)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


def get_supabase() -> Client:
    if not settings.supabase_url or not settings.supabase_service_role_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase not configured",
        )
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


def get_current_user_id(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    supabase: Annotated[Client, Depends(get_supabase)],
) -> str | None:
    """Return auth.uid() from Supabase JWT if present; else None."""
    if not credentials:
        return None
    try:
        user = supabase.auth.get_user(credentials.credentials)
        if user and user.user:
            return str(user.user.id)
        return None
    except Exception:
        return None


def require_user(user_id: Annotated[str | None, Depends(get_current_user_id)]) -> str:
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    return user_id
