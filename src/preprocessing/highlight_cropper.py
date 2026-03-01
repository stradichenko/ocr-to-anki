#!/usr/bin/env python3
"""
Adaptive Highlight Cropper
Detects and crops colored highlights from images with context-aware color detection.
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ColorRange:
    """Represents HSV color range for detection"""
    name: str
    hue_center: float  # Center hue value (0-180 in OpenCV)
    hue_range: float   # Range around center
    sat_min: float     # Minimum saturation (0-255)
    sat_max: float     # Maximum saturation (0-255)
    val_min: float     # Minimum value/brightness (0-255)
    val_max: float     # Maximum value/brightness (0-255)

class AdaptiveHighlightCropper:
    """Crops colored highlights from images with adaptive color detection"""
    
    # Base color definitions in HSV (OpenCV scale: H:0-180, S:0-255, V:0-255)
    # Adjusted for better highlight detection with stricter ranges
    BASE_COLORS = {
        'yellow': ColorRange('yellow', 25, 15, 80, 255, 120, 255),  # Stricter yellow
        'orange': ColorRange('orange', 12, 10, 100, 255, 120, 255),  # Stricter orange
        'red': ColorRange('red', 0, 10, 100, 255, 100, 255),  # Stricter red
        'green': ColorRange('green', 55, 20, 60, 255, 100, 255),  # Stricter green
        'blue': ColorRange('blue', 105, 20, 60, 255, 100, 255),  # Stricter blue
        'purple': ColorRange('purple', 135, 20, 60, 255, 100, 255),  # Stricter purple
    }
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize with configuration"""
        self.config = self._load_config(config_path)
        self.setup_directories()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('highlight_detection', {})
    
    def setup_directories(self):
        """Create necessary directories"""
        self.input_dir = Path(self.config.get('input_dir', 'data/images'))
        self.output_dir = Path(self.config.get('output_dir', 'data/cropped_highlights'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create color subdirectories if organizing by color
        if self.config.get('organize_by_color', True):
            for color in self.BASE_COLORS.keys():
                if self.config.get('colors_to_detect', {}).get(color, True):
                    (self.output_dir / color).mkdir(exist_ok=True)
    
    def analyze_image_context(self, image: np.ndarray) -> Dict:
        """Analyze image lighting and color context"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate image statistics
        mean_hsv = cv2.mean(hsv)[:3]
        std_hsv = cv2.meanStdDev(hsv)[1].flatten()[:3]
        
        # Determine overall brightness and color cast
        context = {
            'mean_hue': mean_hsv[0],
            'mean_sat': mean_hsv[1],
            'mean_val': mean_hsv[2],
            'std_hue': std_hsv[0],
            'std_sat': std_hsv[1],
            'std_val': std_hsv[2],
            'is_low_light': mean_hsv[2] < 100,
            'is_high_saturation': mean_hsv[1] > 150,
            'color_cast': self._detect_color_cast(mean_hsv[0])
        }
        
        logger.debug(f"Image context: {context}")
        return context
    
    def _detect_color_cast(self, mean_hue: float) -> str:
        """Detect dominant color cast in image"""
        # Map hue to color cast
        if mean_hue < 20 or mean_hue > 160:
            return 'red'
        elif 20 <= mean_hue < 40:
            return 'orange'
        elif 40 <= mean_hue < 80:
            return 'yellow'
        elif 80 <= mean_hue < 130:
            return 'green'
        elif 130 <= mean_hue < 160:
            return 'blue'
        else:
            return 'neutral'
    
    def adapt_color_range(self, base_range: ColorRange, context: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Adapt color range based on image context"""
        if not self.config.get('adaptive_mode', True):
            # Use base ranges without adaptation
            lower = np.array([
                base_range.hue_center - base_range.hue_range,
                base_range.sat_min,
                base_range.val_min
            ], dtype=np.uint8)
            upper = np.array([
                base_range.hue_center + base_range.hue_range,
                base_range.sat_max,
                base_range.val_max
            ], dtype=np.uint8)
        else:
            # Adaptive adjustments
            hue_shift = 0
            sat_adjust = 0
            val_adjust = 0
            
            # Adjust for color cast
            if context['color_cast'] != 'neutral':
                # Shift hue range slightly towards the cast
                cast_hue = {'red': 0, 'orange': 20, 'yellow': 50, 
                           'green': 90, 'blue': 120}[context['color_cast']]
                hue_shift = (cast_hue - context['mean_hue']) * 0.1
            
            # Adjust for lighting conditions
            if context['is_low_light']:
                val_adjust = -30  # Lower value threshold
                sat_adjust = -20  # Lower saturation threshold
            
            # Adjust for high saturation environments
            if context['is_high_saturation']:
                sat_adjust = 20  # Increase saturation threshold
            
            # Apply adjustments with configured tolerance
            tolerance = self.config.get('color_tolerance', 30)
            
            # Calculate adjusted values and ensure they're within valid ranges
            lower_hue = int(max(0, min(180, base_range.hue_center - tolerance + hue_shift)))
            lower_sat = int(max(0, min(255, base_range.sat_min + sat_adjust)))
            lower_val = int(max(0, min(255, base_range.val_min + val_adjust)))
            
            upper_hue = int(max(0, min(180, base_range.hue_center + tolerance + hue_shift)))
            upper_sat = int(max(0, min(255, base_range.sat_max)))
            upper_val = int(max(0, min(255, base_range.val_max)))
            
            lower = np.array([lower_hue, lower_sat, lower_val], dtype=np.uint8)
            upper = np.array([upper_hue, upper_sat, upper_val], dtype=np.uint8)
        
        return lower, upper
    
    def detect_color_regions(self, image: np.ndarray, color_name: str, 
                            context: Dict) -> List[np.ndarray]:
        """Detect regions of specified color in image"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        base_range = self.BASE_COLORS[color_name]
        
        # Special handling for red (wraps around in HSV)
        if color_name == 'red':
            # Red lower range (0-10)
            lower1 = np.array([0, int(base_range.sat_min), int(base_range.val_min)], dtype=np.uint8)
            upper1 = np.array([10, int(base_range.sat_max), int(base_range.val_max)], dtype=np.uint8)
            
            # Red upper range (170-180)
            lower2 = np.array([170, int(base_range.sat_min), int(base_range.val_min)], dtype=np.uint8)
            upper2 = np.array([180, int(base_range.sat_max), int(base_range.val_max)], dtype=np.uint8)
            
            # Apply adaptive adjustments if enabled
            if self.config.get('adaptive_mode', True):
                sat_adjust = 0
                val_adjust = 0
                
                if context['is_low_light']:
                    val_adjust = -20
                    sat_adjust = -15
                
                if context['is_high_saturation']:
                    sat_adjust = 15
                
                # Apply adjustments to both ranges
                lower1[1] = int(max(30, min(255, lower1[1] + sat_adjust)))  # Don't go below 30 sat
                lower1[2] = int(max(80, min(255, lower1[2] + val_adjust)))  # Don't go below 80 val
                lower2[1] = int(max(30, min(255, lower2[1] + sat_adjust)))
                lower2[2] = int(max(80, min(255, lower2[2] + val_adjust)))
            
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
            
            logger.debug(f"Red detection - Range1: {lower1} to {upper1}, Range2: {lower2} to {upper2}")
        else:
            lower, upper = self.adapt_color_range(base_range, context)
            mask = cv2.inRange(hsv, lower, upper)
            logger.debug(f"{color_name} detection - Range: {lower} to {upper}")
        
        # Apply morphological operations to clean up mask
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_medium = np.ones((5, 5), np.uint8)
        
        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
        # Connect nearby regions
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
        
        # Save mask for debugging
        if self.config.get('save_visualization', True):
            debug_path = self.output_dir / "debug"
            debug_path.mkdir(exist_ok=True)
            mask_path = debug_path / f"mask_{color_name}.png"
            cv2.imwrite(str(mask_path), mask)
            logger.debug(f"Saved mask for {color_name}: {mask_path}")
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logger.debug(f"Found {len(contours)} raw contours for {color_name}")
        
        # Filter by minimum area
        min_area = self.config.get('min_area', 100)
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        logger.debug(f"After area filtering: {len(valid_contours)} contours for {color_name}")
        
        # Additional filtering: check aspect ratio and solidity to reduce false positives
        filtered_contours = []
        for contour in valid_contours:
            # Calculate aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Calculate solidity (contour area / convex hull area)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = cv2.contourArea(contour) / hull_area if hull_area > 0 else 0
            
            # Filter out contours that are too thin/tall or have very low solidity
            # Highlights are typically somewhat rectangular
            if 0.1 < aspect_ratio < 10 and solidity > 0.3:
                filtered_contours.append(contour)
            else:
                logger.debug(f"Filtered out contour: aspect_ratio={aspect_ratio:.2f}, solidity={solidity:.2f}")
        
        logger.debug(f"After shape filtering: {len(filtered_contours)} contours for {color_name}")
        
        # Merge nearby contours if enabled
        if self.config.get('merge_nearby', True) and len(filtered_contours) > 1:
            filtered_contours = self.merge_nearby_contours(filtered_contours)
            logger.debug(f"After merging: {len(filtered_contours)} contours for {color_name}")
        
        return filtered_contours
    
    def merge_nearby_contours(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """Merge contours that are close to each other"""
        merge_distance = self.config.get('merge_distance', 20)
        merged = []
        used = set()
        
        for i, c1 in enumerate(contours):
            if i in used:
                continue
                
            # Start with current contour
            combined = c1
            used.add(i)
            
            # Check for nearby contours
            x1, y1, w1, h1 = cv2.boundingRect(c1)
            
            for j, c2 in enumerate(contours[i+1:], i+1):
                if j in used:
                    continue
                    
                x2, y2, w2, h2 = cv2.boundingRect(c2)
                
                # Calculate distance between bounding boxes
                x_dist = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
                y_dist = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
                distance = np.sqrt(x_dist**2 + y_dist**2)
                
                if distance <= merge_distance:
                    # Merge contours
                    combined = np.vstack([combined, c2])
                    used.add(j)
                    # Update bounding box for merged region
                    x1 = min(x1, x2)
                    y1 = min(y1, y2)
                    w1 = max(x1 + w1, x2 + w2) - x1
                    h1 = max(y1 + h1, y2 + h2) - y1
            
            merged.append(combined)
        
        return merged
    
    def crop_regions(self, image: np.ndarray, contours: List[np.ndarray], 
                     color_name: str, image_name: str):
        """Crop and save detected regions"""
        padding = self.config.get('padding', 10)
        height, width = image.shape[:2]
        
        for idx, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            
            # Add padding (ensuring we stay within image bounds)
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(width, x + w + padding)
            y_end = min(height, y + h + padding)
            
            # Crop region
            cropped = image[y_start:y_end, x_start:x_end]
            
            # Generate output filename
            base_name = Path(image_name).stem
            output_name = f"{base_name}_{color_name}_{idx:03d}.{self.config.get('output_format', 'png')}"
            
            # Determine output path
            if self.config.get('organize_by_color', True):
                output_path = self.output_dir / color_name / output_name
            else:
                output_path = self.output_dir / output_name
            
            # Save cropped region
            cv2.imwrite(str(output_path), cropped)
            logger.info(f"Saved: {output_path}")
    
    def save_visualization(self, image: np.ndarray, all_contours: Dict[str, List], 
                          image_name: str):
        """Save visualization of detected regions"""
        vis_image = image.copy()
        
        # Color map for visualization
        color_map = {
            'yellow': (0, 255, 255),
            'orange': (0, 165, 255),
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'purple': (255, 0, 255)
        }
        
        # Draw all detected regions
        for color_name, contours in all_contours.items():
            color = color_map.get(color_name, (255, 255, 255))
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(vis_image, color_name, (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Save visualization
        base_name = Path(image_name).stem
        vis_path = self.output_dir / f"{base_name}_visualization.png"
        cv2.imwrite(str(vis_path), vis_image)
        logger.info(f"Saved visualization: {vis_path}")
    
    def process_image(self, image_path: Path):
        """Process a single image"""
        logger.info(f"Processing: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return
        
        logger.info(f"Image shape: {image.shape}")
        
        # Analyze image context
        context = self.analyze_image_context(image)
        logger.info(f"Image context - Mean HSV: ({context['mean_hue']:.1f}, {context['mean_sat']:.1f}, {context['mean_val']:.1f})")
        logger.info(f"Color cast: {context['color_cast']}, Low light: {context['is_low_light']}")
        
        # Detect and crop highlights for each color
        all_contours = {}
        colors_config = self.config.get('colors_to_detect', {})
        
        # Get list of colors to actually process
        colors_to_process = [
            color for color, enabled in colors_config.items() 
            if enabled and color in self.BASE_COLORS
        ]
        
        if not colors_to_process:
            logger.warning("No colors enabled for detection in config")
            return
        
        logger.info(f"Detecting colors: {', '.join(colors_to_process)}")
        
        total_regions = 0
        for color_name in colors_to_process:
            logger.info(f"Searching for {color_name} highlights...")
            contours = self.detect_color_regions(image, color_name, context)
            if contours:
                all_contours[color_name] = contours
                self.crop_regions(image, contours, color_name, image_path.name)
                logger.info(f"✓ Found {len(contours)} {color_name} regions")
                total_regions += len(contours)
            else:
                logger.info(f"✗ No {color_name} regions found")
        
        if total_regions == 0:
            logger.warning(f"No highlighted regions found in {image_path.name}")
            logger.info("Try adjusting detection parameters in settings.yaml:")
            logger.info("  - Increase color_tolerance (currently {})".format(self.config.get('color_tolerance', 25)))
            logger.info("  - Lower min_area (currently {})".format(self.config.get('min_area', 50)))
            logger.info("  - Enable adaptive_mode if lighting varies")
        else:
            logger.info(f"Total regions found: {total_regions}")
        
        # Save visualization if enabled and regions were found
        if self.config.get('save_visualization', True):
            self.save_visualization(image, all_contours, image_path.name)
    
    def process_all(self):
        """Process all images in input directory"""
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.input_dir.glob(f'*{ext}'))
            image_files.extend(self.input_dir.glob(f'*{ext.upper()}'))
        
        if not image_files:
            logger.warning(f"No images found in {self.input_dir}")
            return
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Process each image
        for image_path in image_files:
            try:
                self.process_image(image_path)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                continue
        
        logger.info("Processing complete!")

def main():
    """Main entry point"""
    import sys
    
    # Enable debug logging if --debug flag is passed
    if '--debug' in sys.argv:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
    
    cropper = AdaptiveHighlightCropper()
    cropper.process_all()

if __name__ == "__main__":
    main()
