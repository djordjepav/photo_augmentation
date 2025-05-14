from datetime import datetime

from fastapi import FastAPI
from tinydb import TinyDB, Query

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

app = FastAPI()
db = TinyDB(f"./tiny_db/db_{timestamp}.json")
