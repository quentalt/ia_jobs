import os
from fastapi import FastAPI, HTTPException, Depends, Query, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, Table, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from passlib.context import CryptContext
from jose import JWTError, jwt
from transformers import pipeline
from typing import List, Optional
from datetime import datetime, timedelta
import re

# Configuration de la base de données PostgreSQL
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/jobs")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Configuration des constantes JWT
SECRET_KEY = os.getenv("SECRET_KEY", "secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Configuration de l'outil de hachage des mots de passe
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Configuration de l'authentification
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Modèle de sentiment analysis

sentiment_analyzer = pipeline('sentiment-analysis',model='distilbert-base-uncased-finetuned-sst-2-english')

# Définition de la structure des tables
user_skill_association = Table(
    "user_skill_association",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id")),
    Column("skill_id", Integer, ForeignKey("skills.id"))
)

job_skill_association = Table(
    "job_skill_association",
    Base.metadata,
    Column("job_id", Integer, ForeignKey("jobs.id")),
    Column("skill_id", Integer, ForeignKey("skills.id"))
)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    skills = relationship("Skill", secondary=user_skill_association, back_populates="users")


class Skill(Base):
    __tablename__ = "skills"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    users = relationship("User", secondary=user_skill_association, back_populates="skills")
    jobs = relationship("Job", secondary=job_skill_association, back_populates="skills")


class Job(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(Text)
    skills = relationship("Skill", secondary=job_skill_association, back_populates="jobs")


class Favorite(Base):
    __tablename__ = "favorites"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    job_id = Column(Integer, ForeignKey('jobs.id'))
    user = relationship("User", back_populates="favorites")
    job = relationship("Job", back_populates="favorited_by")


User.favorites = relationship("Favorite", back_populates="user")
Job.favorited_by = relationship("Favorite", back_populates="job")

# Créer les tables et les index
Base.metadata.create_all(bind=engine)
Index('ix_jobs_title', Job.title)
Index('ix_jobs_description', Job.description)


# Définir les modèles de données
class UserCreate(BaseModel):
    username: str
    password: str
    skills: List[str] = []


class UserUpdate(BaseModel):
    skills: List[str] = []


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class JobCreate(BaseModel):
    title: str
    description: str
    skills: List[str] = []


class JobResponse(BaseModel):
    id: int
    title: str
    description: str
    sentiment: str
    skills: List[str]


class PaginatedJobsResponse(BaseModel):
    jobs: List[JobResponse]
    total_pages: int


# Initialiser l'application FastAPI
app = FastAPI()

# Configurer les CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Fonction pour obtenir une session de base de données
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Fonctions utilitaires pour la gestion des utilisateurs
def get_password_hash(password):
    return pwd_context.hash(password)


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_user(db, username: str):
    return db.query(User).filter(User.username == username).first()


def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user


def get_current_user(token: str = Depends(oauth2_scheme), db: SessionLocal = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


def get_or_create_skill(db, skill_name: str):
    skill = db.query(Skill).filter(Skill.name == skill_name).first()
    if not skill:
        skill = Skill(name=skill_name)
        db.add(skill)
        db.commit()
        db.refresh(skill)
    return skill


@app.post("/token", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: SessionLocal = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/users/", response_model=Token)
def create_user(user: UserCreate, db: SessionLocal = Depends(get_db)):
    hashed_password = get_password_hash(user.password)
    db_user = User(username=user.username, hashed_password=hashed_password)
    for skill_name in user.skills:
        skill = get_or_create_skill(db, skill_name)
        db_user.skills.append(skill)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": db_user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.put("/users/me", response_model=UserUpdate)
def update_user(user_update: UserUpdate, db: SessionLocal = Depends(get_db),
                current_user: User = Depends(get_current_user)):
    current_user.skills = []
    for skill_name in user_update.skills:
        skill = get_or_create_skill(db, skill_name)
        current_user.skills.append(skill)
    db.commit()
    db.refresh(current_user)
    return user_update


@app.put("/jobs/{job_id}", response_model=JobResponse)
def update_job(job_id: int, job: JobCreate, db: SessionLocal = Depends(get_db),
               current_user: User = Depends(get_current_user)):
    db_job = db.query(Job).filter(Job.id == job_id).first()
    if not db_job:
        raise HTTPException(status_code=404, detail="Job not found")
    db_job.title = job.title
    db_job.description = job.description
    db_job.skills = []
    for skill_name in job.skills:
        skill = get_or_create_skill(db, skill_name)
        db_job.skills.append(skill)
    db.commit()
    db.refresh(db_job)
    sentiment = sentiment_analyzer(job.description)[0]
    return {
        "id": db_job.id,
        "title": db_job.title,
        "description": db_job.description,
        "sentiment": sentiment['label'],
        "skills": [skill.name for skill in db_job.skills]
    }


@app.post("/jobs/", response_model=JobResponse)
def create_job(job: JobCreate, db: SessionLocal = Depends(get_db), current_user: User = Depends(get_current_user)):
    db_job = Job(title=job.title, description=job.description)
    for skill_name in job.skills:
        skill = get_or_create_skill(db, skill_name)
        db_job.skills.append(skill)
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    sentiment = sentiment_analyzer(job.description)[0]
    return {
        "id": db_job.id,
        "title": db_job.title,
        "description": db_job.description,
        "sentiment": sentiment['label'],
        "skills": [skill.name for skill in db_job.skills]
    }


@app.get("/jobs/", response_model=PaginatedJobsResponse)
def read_jobs(skip: int = 0, limit: int = 10, title: Optional[str] = None, sentiment: Optional[str] = None,
              skill: Optional[str] = None, db: SessionLocal = Depends(get_db),
              current_user: User = Depends(get_current_user)):
    query = db.query(Job)
    if title:
        query = query.filter(Job.title.contains(title))
    if sentiment:
        query = query.filter(Job.description.contains(sentiment))
    if skill:
        query = query.join(Job.skills).filter(Skill.name == skill)
    total_jobs = query.count()
    jobs = query.offset(skip).limit(limit).all()
    total_pages = (total_jobs + limit - 1) // limit
    job_responses = []
    for job in jobs:
        sentiment = sentiment_analyzer(job.description)[0]
        job_responses.append({
            "id": job.id,
            "title": job.title,
            "description": job.description,
            "sentiment": sentiment['label'],
            "skills": [skill.name for skill in job.skills]
        })
    return PaginatedJobsResponse(jobs=job_responses, total_pages=total_pages)


@app.get("/jobs/recommendations/", response_model=PaginatedJobsResponse)
def recommend_jobs(skip: int = 0, limit: int = 10, db: SessionLocal = Depends(get_db),
                   current_user: User = Depends(get_current_user)):
    user_skills = [skill.name for skill in current_user.skills]
    query = db.query(Job).join(Job.skills).filter(Skill.name.in_(user_skills)).distinct()
    total_jobs = query.count()
    jobs = query.offset(skip).limit(limit).all()
    total_pages = (total_jobs + limit - 1) // limit
    job_responses = []
    for job in jobs:
        sentiment = sentiment_analyzer(job.description)[0]
        job_responses.append({
            "id": job.id,
            "title": job.title,
            "description": job.description,
            "sentiment": sentiment['label'],
            "skills": [skill.name for skill in job.skills]
        })
    return PaginatedJobsResponse(jobs=job_responses, total_pages=total_pages)
