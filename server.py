from fastapi import FastAPI
import uvicorn
import identifier_insect

app = FastAPI()


@app.get('/index')
def hello():
    return f"Hello!"


@app.post('/predict')
async def predict_image(data: dict):
    name_insect =  identifier_insect.IdentifierInsect.identifier(data['id'],data['image_url'])
    return f"{name_insect}"


if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='0.0.0.0')
