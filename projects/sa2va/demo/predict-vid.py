from cog import BasePredictor, Input, Path, BaseModel
import os
import cv2
import time
import shutil
import subprocess
import numpy as np
from PIL import Image
import tempfile
from transformers import AutoModelForCausalLM, AutoTokenizer
from mmengine.visualization import Visualizer
from typing import Optional
from third_parts import VideoReader

MODEL_CACHE = "checkpoints"
# MODEL_URL = "https://weights.replicate.delivery/default/ByteDance/Sa2VA-4B/model.tar"
MODEL_URL = "https://weights.replicate.delivery/default/ByteDance/Sa2VA-8B/model.tar"
# MODEL_URL = "https://weights.replicate.delivery/default/ByteDance/Sa2VA-26B/model.tar"

class Output(BaseModel):
    masked_video: Optional[Path]
    response: str

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

def read_video(video_path, video_interval):
    # First verify the video can be opened
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    cap.release()

    # Read frames using VideoReader
    vid_frames = VideoReader(video_path)[::video_interval]
    if len(vid_frames) == 0:
        raise ValueError(f"No frames could be read from video: {video_path}")
    
    temp_dir = tempfile.mkdtemp()
    os.makedirs(temp_dir, exist_ok=True)
    image_paths = []
    processed_frames = []
    
    for frame_idx, frame_image in enumerate(vid_frames):
        if frame_image is None:
            continue
            
        # Convert BGR to RGB
        frame_image = frame_image[..., ::-1]  # BGR to RGB
        frame_image = Image.fromarray(frame_image)
        processed_frames.append(frame_image)

        image_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.jpg")
        frame_image.save(image_path, format="JPEG")
        image_paths.append(image_path)
    
    if not processed_frames:
        raise ValueError("No valid frames were processed from the video")
        
    return processed_frames, image_paths

def visualize(pred_mask, image_path, work_dir):
    visualizer = Visualizer()
    img = cv2.imread(image_path)
    visualizer.set_image(img)
    visualizer.draw_binary_masks(pred_mask, colors='g', alphas=0.4)
    visual_result = visualizer.get_image()

    output_path = os.path.join(work_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, visual_result)
    return output_path
    
class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        # Download weights if they don't exist
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_CACHE,
            torch_dtype="auto",
            device_map="cuda:0",
            trust_remote_code=True,
        ).eval().cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CACHE,
            trust_remote_code=True,
        )

    def predict(
        self,
        video: Path = Input(description="Input video for segmentation"),
        instruction: str = Input(description="Text instruction for the model"),
        frame_interval: int = Input(description="Frame interval for processing", default=6, ge=1, le=30),
    ) -> Output:
        """Run a single prediction on the model"""
        # clean up past runs remove /tmp/output folder
        if os.path.exists("/tmp/output"):
            shutil.rmtree("/tmp/output")
        
        os.makedirs("/tmp/output")

        # Process video frames
        vid_frames, image_paths = read_video(str(video), frame_interval)
        
        # Get video properties for output
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise ValueError("Failed to open video file")
        original_fps = cap.get(cv2.CAP_PROP_FPS) 
        if original_fps == 0:
            original_fps = 30.0  # Default to 30fps if unable to read
        new_fps = original_fps / frame_interval if frame_interval > 1 else original_fps
        cap.release()

        # Prepare the input
        question = f"<image>{instruction}"
        result = self.model.predict_forward(
            video=vid_frames,
            text=question,
            tokenizer=self.tokenizer,
        )
        prediction = result['prediction']

        output_video_path = None
        masked_video_path = None

        if '[SEG]' in prediction:
            _seg_idx = 0
            pred_masks = result['prediction_masks'][_seg_idx]
            seg_frames = []
            masked_only_frames = []

            temp_dir = tempfile.mkdtemp()
            os.makedirs(temp_dir, exist_ok=True)

            # Process each frame
            for frame_idx in range(len(vid_frames)):
                pred_mask = pred_masks[frame_idx]
                
                # Create visualized frame with segmentation overlay
                seg_frame = visualize(pred_mask, image_paths[frame_idx], temp_dir)
                seg_frames.append(seg_frame)

                # Create binary mask frame
                binary_mask = (pred_mask.astype('uint8') * 255)
                binary_mask_path = os.path.join(temp_dir, f"binary_mask_{frame_idx}.png")
                cv2.imwrite(binary_mask_path, binary_mask)
                masked_only_frames.append(binary_mask_path)

            # Read first frame for dimensions
            frame = cv2.imread(seg_frames[0])
            height, width, layers = frame.shape

            # Create output video files
            masked_video_path = "/tmp/output/masked_video.mp4"
            temp_masked_path = "/tmp/output/temp_masked.avi"

            # Define video writer using a more basic codec
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            masked_video_writer = cv2.VideoWriter(temp_masked_path, fourcc, new_fps, (width, height), isColor=False)

            # Write frames to video
            for mask_frame_path in masked_only_frames:
                mask_frame = cv2.imread(mask_frame_path, cv2.IMREAD_GRAYSCALE)
                masked_video_writer.write(mask_frame)

            # Release video writer
            masked_video_writer.release()

            # Convert to web-compatible MP4 using ffmpeg
            subprocess.run([
                'ffmpeg', '-i', temp_masked_path, '-c:v', 'libx264',
                '-preset', 'fast', '-pix_fmt', 'yuv420p', masked_video_path
            ], check=True)

            # Clean up temporary file
            os.remove(temp_masked_path)

        return Output(
            masked_video=Path(masked_video_path) if masked_video_path else None,
            response=str(prediction)
        ) 