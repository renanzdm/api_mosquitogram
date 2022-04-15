from pydantic import BaseModel


class IdIdentifyModel(BaseModel):
    id: str
    image_url: str 

