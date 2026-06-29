import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class DBMarket(Base):
    __tablename__ = "markets"

    id = Column(Integer, primary_key=True, index=True)
    asset = Column(String, nullable=False)
    platform = Column(String, nullable=False)
    question = Column(String, nullable=False)
    market_probability = Column(Float, nullable=False)
    expiration = Column(String, nullable=True)
    market_id = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class DBPrediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False)
    interval = Column(String, nullable=False)
    prediction_direction = Column(String, nullable=False)  # UP / DOWN
    confidence = Column(Float, nullable=False)
    model_probability = Column(Float, nullable=False)
    market_probability = Column(Float, nullable=True)
    kelly_fraction = Column(Float, nullable=True)
    reasoning = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    feedbacks = relationship("DBFeedback", back_populates="prediction", cascade="all, delete-orphan")

class DBFeedback(Base):
    __tablename__ = "feedbacks"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), unique=True, nullable=False)
    actual_movement = Column(String, nullable=False)  # UP / DOWN
    correct = Column(Boolean, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    prediction = relationship("DBPrediction", back_populates="feedbacks")

class DBStatistics(Base):
    __tablename__ = "statistics"

    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String, unique=True, nullable=False)
    metric_value = Column(Float, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
