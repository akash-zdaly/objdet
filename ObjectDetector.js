import {
  media,
  MobileModel,
  torch,
  torchvision,
} from 'react-native-pytorch-core';
import COCO_CLASSES from './CoCoClasses.json';

const T = torchvision.transforms;

// const MODEL_URL =
//   "https://github.com/pytorch/live/releases/download/v0.1.0/detr_resnet50.ptl";

const MODEL_URL =
  'https://github.com/akash-zdaly/ImgDet/raw/main/assests/yolov5s.ptl';

let model = null;
const probabilityThreshold = 0.7;
console.log(MODEL_URL);
export default async function detectObjects(image) {
  // Get image width and height
  const imageWidth = image.getWidth();
  const imageHeight = image.getHeight();

  // Convert image to blob, which is a byte representation of the image
  // in the format height (H), width (W), and channels (C), or HWC for short
  const blob = media.toBlob(image);

  // Get a tensor from image the blob and also define in what format
  // the image blob is.
  let tensor = torch.fromBlob(blob, [imageHeight, imageWidth, 3]);

  // Rearrange the tensor shape to be [CHW]
  tensor = tensor.permute([2, 0, 1]);

  // Divide the tensor values by 255 to get values between [0, 1]
  tensor = tensor.div(255);

  // Resize the image tensor to 3 x min(height, 800) x min(width, 800)
  const resize = T.resize(800);
  tensor = resize(tensor);

  // Normalize the tensor image with mean and standard deviation
  const normalize = T.normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
  tensor = normalize(tensor);

  // Unsqueeze adds 1 leading dimension to the tensor
  const formattedInputTensor = tensor.unsqueeze(0);

  // Load model if not loaded
  if (model == null) {
    console.log('Loading model...');
    const filePath = await MobileModel.download(MODEL_URL);
    model = await torch.jit._loadForMobile(filePath);
    console.log('Model successfully loaded');
  }

  // Run inference
  const output = await model.forward(formattedInputTensor);

  const predLogits = output.pred_logits.squeeze(0);
  const predBoxes = output.pred_boxes.squeeze(0);

  const numPredictions = predLogits.shape[0];

  const resultBoxes = [];

  for (let i = 0; i < numPredictions; i++) {
    const confidencesTensor = predLogits[i];
    const scores = confidencesTensor.softmax(0);
    const maxIndex = confidencesTensor.argmax().item();
    const maxProb = scores[maxIndex].item();

    if (maxProb <= probabilityThreshold || maxIndex >= COCO_CLASSES.length) {
      continue;
    }

    const boxTensor = predBoxes[i];
    const [centerX, centerY, boxWidth, boxHeight] = boxTensor.data();
    const x = centerX - boxWidth / 2;
    const y = centerY - boxHeight / 2;

    // Adjust bounds to image size
    const bounds = [
      x * imageWidth,
      y * imageHeight,
      boxWidth * imageWidth,
      boxHeight * imageHeight,
    ];

    const match = {
      objectClass: COCO_CLASSES[maxIndex],
      bounds,
    };

    resultBoxes.push(match);
  }

  return resultBoxes;
}
