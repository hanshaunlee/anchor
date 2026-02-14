from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    supabase_url: str = ""
    supabase_service_role_key: str = ""
    database_url: str = ""
    jwt_secret: str = ""
    # Neo4j (optional): visualization/investigation only; not used by ML pipeline
    neo4j_uri: str = ""
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
