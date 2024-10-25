from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes
import cv2
import numpy as np
from collections import deque
import dotenv
import os
import sys

dotenv.load_dotenv()
running = True


def custom_on_prediction(predictions, frame):
    global running
    try:
        render_boxes(predictions, frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            return False

    except Exception as e:
        print(f"Error in custom_on_prediction: {e}")



try:
    pipeline = InferencePipeline.init(
        model_id="living-room-items/1",  # Replace with your actual model ID
        video_reference=0,  # Adjust if needed for your camera setup
        api_key= os.environ.get("ROBOFLOW_API"),
        on_prediction=custom_on_prediction,
    )

    pipeline.start()

    while running and pipeline.is_alive():
        pipeline.join(timeout=0.1)
        
    pipeline.join()
except Exception as e:
    print(f"Error initializing or running pipeline: {e}")
finally:
    cv2.destroyAllWindows()
    sys.exit(0)