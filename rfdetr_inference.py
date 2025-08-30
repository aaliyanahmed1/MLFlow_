import io
import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='RFDeTR Object Detection')
    parser.add_argument('--image_url', type=str, default="images/sample.png",
                        help='URL of the image to perform detection on')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection confidence threshold')
    parser.add_argument('--optimize', action='store_true',
                        help='Optimize model for inference')
    parser.add_argument('--output', type=str, default="detection_output.jpg",
                        help='Output image filename')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("Loading RFDeTR model...")
    model = RFDETRBase()
    
    if args.optimize:
        print("Optimizing model for inference...")
        model.optimize_for_inference()
    
    print(f"Downloading image from {args.image_url}...")
    try:
        image = Image.open(io.BytesIO(requests.get(args.image_url).content))
    except Exception as e:
        print(f"Error downloading image: {e}")
        return
    
    print(f"Running inference with threshold {args.threshold}...")
    detections = model.predict(image, threshold=args.threshold)
    
    # Create labels for visualization
    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]
    
    # Convert PIL Image to numpy array for supervision visualization
    image_np = sv.Image.from_pil(image)
    
    # Create box annotator
    box_annotator = sv.BoxAnnotator()
    
    # Annotate image
    annotated_image = box_annotator.annotate(
        scene=image_np.copy(),
        detections=detections,
        labels=labels
    )
    
    # Save and display result
    plt.figure(figsize=(10, 10))
    plt.imshow(annotated_image)
    plt.axis('off')
    plt.savefig(args.output, bbox_inches='tight')
    plt.close()
    
    print(f"Detection completed. Found {len(detections)} objects.")
    print(f"Results saved to {args.output}")
    
    # Print detection results
    for i, (class_id, confidence, bbox) in enumerate(zip(
            detections.class_id, detections.confidence, detections.xyxy)):
        print(f"Detection {i+1}: {COCO_CLASSES[class_id]} (Confidence: {confidence:.2f})")
        print(f"  Bounding box: {bbox}")

if __name__ == "__main__":
    main()