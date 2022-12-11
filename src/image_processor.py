from utils import display_instances, rotate_bbox
from config import config
from torchvision.transforms import functional as F


class ImageRunner:
    def __init__(self, model, img, threshold=0.7):
        self.image = img
        self.model = model
        self.threshold = threshold

    def process_image(self):
        """
         Classifies the shape of cloth
         """
        img_tensor = F.to_tensor(self.image)
        result = self.model([img_tensor.to(config['DEVICE'])])[0]

        masks = result['masks'].squeeze().permute(1, 2, 0).cpu().numpy().round()
        boxes = result['boxes'].cpu().numpy()
        scores = result['scores'].cpu().numpy()
        class_ids = result['labels'].cpu().numpy()
        class_names = {i: config['labels'][i - 1] for i in class_ids}

        # Adjust bboxes to image.
        boxes = rotate_bbox(boxes, 270, (362, 562))

        display_instances(self.image, boxes, masks, class_ids, class_names, scores=scores, confidence=self.threshold)
        return 'result.png'
