from app.database import engine
from app.models import Base

def reset_database():
    # Drop all tables
    Base.metadata.drop_all(bind=engine)
    # Create all tables
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    reset_database()
    print("Database reset complete!") 