from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import declarative_base, sessionmaker

# 1. Create the SQLite Database File
SQLALCHEMY_DATABASE_URL = "sqlite:///./saas_platform.db"

# 2. Establish the Connection Engine
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 3. Create the Base Class for Models
Base = declarative_base()

# 4. Define a Sample Table (e.g., User/Client Profile)
class Client(Base):
    __tablename__ = "clients"
    
    id = Column(Integer, primary_key=True, index=True)
    company_name = Column(String, unique=True, index=True)
    industry = Column(String)
    # E.g., The copper price where this company starts losing money
    copper_opex_threshold = Column(Float, nullable=True) 

# 5. Generate the Database
Base.metadata.create_all(bind=engine)