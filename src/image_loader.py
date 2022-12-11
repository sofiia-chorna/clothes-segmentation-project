from imgurpython import ImgurClient
import requests
from io import BytesIO
from PIL import Image

class ImageLoader:
    def __init__(self, client_id, client_secret):
        self.client = ImgurClient(client_id, client_secret)

    def get_image(self, image_name):
        item = self.client.get_image(image_name)
        response = requests.get(item.link)
        return Image.open(BytesIO(response.content))
