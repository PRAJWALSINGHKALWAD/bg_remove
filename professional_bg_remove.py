"""
Professional Background Removal System
Advanced multi-model AI-powered background removal with intelligent subject detection
"""

from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import cv2
from rembg import remove, new_session
import os
import sys
import argparse
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelConfig:
    """Configuration for different AI models"""
    MODELS = {
        'u2net': {
            'name': 'u2net',
            'description': 'General purpose model - good for most objects',
            'best_for': ['general', 'objects', 'mixed'],
            'quality': 'good'
        },
        'u2net_human_seg': {
            'name': 'u2net_human_seg',
            'description': 'Specialized for human segmentation',
            'best_for': ['human', 'person', 'people'],
            'quality': 'excellent'
        },
        'silueta': {
            'name': 'silueta',
            'description': 'High-quality people and portraits',
            'best_for': ['portrait', 'human', 'person'],
            'quality': 'excellent'
        },
        'isnet-general-use': {
            'name': 'isnet-general-use',
            'description': 'High-quality general purpose model',
            'best_for': ['general', 'objects', 'detailed'],
            'quality': 'excellent'
        },
        'u2net_cloth_seg': {
            'name': 'u2net_cloth_seg',
            'description': 'Specialized for clothing items',
            'best_for': ['clothing', 'fashion', 'apparel'],
            'quality': 'excellent'
        }
    }

class SubjectDetector:
    """Intelligent subject detection for optimal model selection"""
    
    def __init__(self):
        self.face_cascade = None
        self._load_classifiers()
    
    def _load_classifiers(self):
        """Load OpenCV classifiers"""
        try:
            # Try to load face detection classifier
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception as e:
            logger.warning(f"Could not load face classifier: {e}")
    
    def detect_subject_type(self, image: Image.Image) -> Dict[str, float]:
        """
        Analyze image to determine subject type and confidence scores
        Returns: Dict with subject types and confidence scores
        """
        results = {
            'human': 0.0,
            'object': 0.0,
            'animal': 0.0,
            'clothing': 0.0,
            'complex': 0.0
        }
        
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array
        
        # Face detection
        if self.face_cascade is not None:
            faces = self.face_cascade.detectMultiScale(img_gray, 1.1, 4)
            if len(faces) > 0:
                results['human'] = min(0.9, 0.3 + len(faces) * 0.2)
        
        # Analyze image characteristics
        results.update(self._analyze_image_features(img_array))
        
        return results
    
    def _analyze_image_features(self, img_array: np.ndarray) -> Dict[str, float]:
        """Analyze image features for subject classification"""
        features = {}
        
        # Color analysis
        if len(img_array.shape) == 3:
            # Check for skin tones (rough heuristic)
            skin_mask = self._detect_skin_tones(img_array)
            skin_ratio = np.sum(skin_mask) / (img_array.shape[0] * img_array.shape[1])
            
            if skin_ratio > 0.1:
                features['human'] = max(features.get('human', 0), skin_ratio * 2)
            
            # Edge complexity analysis
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            if edge_density > 0.3:
                features['complex'] = edge_density
            
            # Color diversity (objects tend to have more uniform colors)
            hist = cv2.calcHist([img_array], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            color_diversity = np.count_nonzero(hist) / hist.size
            
            if color_diversity < 0.3:
                features['object'] = 0.6
            
        return features
    
    def _detect_skin_tones(self, img_array: np.ndarray) -> np.ndarray:
        """Simple skin tone detection"""
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Define skin color ranges in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        return skin_mask

class AdvancedPostProcessor:
    """Advanced post-processing for professional results"""
    
    @staticmethod
    def refine_edges(mask: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Advanced edge refinement using guided filtering"""
        # Convert mask to float
        mask_float = mask.astype(np.float32) / 255.0
        
        # Apply bilateral filter for edge-preserving smoothing
        refined = cv2.bilateralFilter(mask_float, 9, 75, 75)
        
        # Morphological operations for cleanup
        kernel = np.ones((3, 3), np.uint8)
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
        
        return (refined * 255).astype(np.uint8)
    
    @staticmethod
    def enhance_hair_details(mask: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Enhance fine hair details using texture analysis"""
        # Convert to grayscale for texture analysis
        if len(original.shape) == 3:
            gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        else:
            gray = original
        
        # Detect hair-like textures using Gabor filters
        hair_enhanced = AdvancedPostProcessor._apply_gabor_filters(gray, mask)
        
        # Combine with original mask
        enhanced_mask = cv2.addWeighted(mask, 0.7, hair_enhanced, 0.3, 0)
        
        return enhanced_mask
    
    @staticmethod
    def _apply_gabor_filters(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply Gabor filters to detect hair-like textures"""
        # Create multiple Gabor kernels for different orientations
        kernels = []
        for theta in range(0, 180, 30):  # 6 orientations
            kernel = cv2.getGaborKernel((21, 21), 5, np.radians(theta), 2*np.pi*0.5, 0.5, 0, ktype=cv2.CV_32F)
            kernels.append(kernel)
        
        # Apply filters and combine responses
        responses = []
        for kernel in kernels:
            filtered = cv2.filter2D(gray, cv2.CV_8UC1, kernel)
            responses.append(filtered)
        
        # Combine responses
        combined = np.maximum.reduce(responses)
        
        # Apply only to mask regions
        hair_mask = cv2.bitwise_and(combined, combined, mask=mask)
        
        return hair_mask
    
    @staticmethod
    def apply_feathering(mask: np.ndarray, feather_radius: int = 3) -> np.ndarray:
        """Apply soft feathering to mask edges"""
        # Create distance transform
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # Normalize and apply feathering
        max_dist = feather_radius
        feathered = np.clip(dist_transform / max_dist, 0, 1)
        
        # Apply feathering only to edge regions
        edges = cv2.Canny(mask, 50, 150)
        edge_dilated = cv2.dilate(edges, np.ones((feather_radius*2, feather_radius*2), np.uint8))
        
        result = mask.copy().astype(np.float32)
        result[edge_dilated > 0] = feathered[edge_dilated > 0] * 255
        
        return result.astype(np.uint8)

class ProfessionalBGRemover:
    """Professional-grade background removal system"""
    
    def __init__(self):
        self.detector = SubjectDetector()
        self.processor = AdvancedPostProcessor()
        self.models_cache = {}
        
    def auto_remove_background(self, 
                             image_path: str, 
                             output_path: str = None,
                             quality: str = 'ultra',
                             auto_detect: bool = True,
                             model_override: str = None,
                             enhance_details: bool = True,
                             feather_edges: bool = True) -> str:
        """
        Professional background removal with automatic optimization
        
        Args:
            image_path: Input image path
            output_path: Output path (auto-generated if None)
            quality: 'good', 'high', 'ultra'
            auto_detect: Use intelligent model selection
            model_override: Force specific model
            enhance_details: Apply detail enhancement
            feather_edges: Apply edge feathering
        """
        
        logger.info(f"Processing {image_path} with {quality} quality")
        
        # Load image
        try:
            original = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Cannot load image: {e}")
        
        # Generate output path if not provided
        if not output_path:
            base_path = Path(image_path)
            output_path = str(base_path.parent / f"{base_path.stem}_professional_nobg.png")
        
        # Select optimal model
        if model_override:
            model_name = model_override
            logger.info(f"Using override model: {model_name}")
        elif auto_detect:
            model_name = self._select_optimal_model(original)
            logger.info(f"Auto-selected model: {model_name}")
        else:
            model_name = 'isnet-general-use'  # Default high-quality model
        
        # Process based on quality setting
        if quality == 'ultra':
            result = self._ultra_quality_process(original, model_name, enhance_details, feather_edges)
        elif quality == 'high':
            result = self._high_quality_process(original, model_name, enhance_details)
        else:
            result = self._standard_process(original, model_name)
        
        # Save result
        result.save(output_path, 'PNG', optimize=True)
        logger.info(f"Saved professional result: {output_path}")
        
        return output_path
    
    def _select_optimal_model(self, image: Image.Image) -> str:
        """Intelligently select the best model for the image"""
        subject_scores = self.detector.detect_subject_type(image)
        
        logger.info(f"Subject detection scores: {subject_scores}")
        
        # Decision logic based on scores
        if subject_scores['human'] > 0.6:
            return 'u2net_human_seg'  # Best for humans
        elif subject_scores['human'] > 0.3:
            return 'silueta'  # Good for portraits
        elif subject_scores['clothing'] > 0.5:
            return 'u2net_cloth_seg'  # Best for clothing
        elif subject_scores['complex'] > 0.4:
            return 'isnet-general-use'  # Best for complex scenes
        else:
            return 'u2net'  # General purpose
    
    def _get_model_session(self, model_name: str):
        """Get cached model session"""
        if model_name not in self.models_cache:
            logger.info(f"Loading model: {model_name}")
            self.models_cache[model_name] = new_session(model_name)
        return self.models_cache[model_name]
    
    def _ultra_quality_process(self, image: Image.Image, model_name: str, 
                              enhance_details: bool, feather_edges: bool) -> Image.Image:
        """Ultra-quality processing with multiple passes and refinements"""
        logger.info("Applying ultra-quality processing...")
        
        # Multi-model ensemble approach
        primary_session = self._get_model_session(model_name)
        primary_result = remove(image, session=primary_session)
        
        # Second pass with different model for refinement
        secondary_model = 'isnet-general-use' if model_name != 'isnet-general-use' else 'u2net'
        secondary_session = self._get_model_session(secondary_model)
        secondary_result = remove(image, session=secondary_session)
        
        # Combine results intelligently
        combined = self._combine_masks(primary_result, secondary_result, image)
        
        if enhance_details:
            combined = self._enhance_details(combined, image)
        
        if feather_edges:
            combined = self._apply_edge_feathering(combined)
        
        return combined
    
    def _high_quality_process(self, image: Image.Image, model_name: str, 
                             enhance_details: bool) -> Image.Image:
        """High-quality processing with refinements"""
        logger.info("Applying high-quality processing...")
        
        session = self._get_model_session(model_name)
        result = remove(image, session=session)
        
        if enhance_details:
            result = self._enhance_details(result, image)
        
        return result
    
    def _standard_process(self, image: Image.Image, model_name: str) -> Image.Image:
        """Standard processing"""
        logger.info("Applying standard processing...")
        
        session = self._get_model_session(model_name)
        return remove(image, session=session)
    
    def _combine_masks(self, primary: Image.Image, secondary: Image.Image, 
                      original: Image.Image) -> Image.Image:
        """Intelligently combine two mask results"""
        # Convert to numpy arrays
        primary_array = np.array(primary)
        secondary_array = np.array(secondary)
        
        # Extract alpha channels
        primary_alpha = primary_array[:, :, 3] if primary_array.shape[2] == 4 else np.ones_like(primary_array[:, :, 0]) * 255
        secondary_alpha = secondary_array[:, :, 3] if secondary_array.shape[2] == 4 else np.ones_like(secondary_array[:, :, 0]) * 255
        
        # Combine using maximum confidence
        combined_alpha = np.maximum(primary_alpha, secondary_alpha)
        
        # Use primary result as base
        result_array = primary_array.copy()
        if result_array.shape[2] == 4:
            result_array[:, :, 3] = combined_alpha
        
        return Image.fromarray(result_array, 'RGBA')
    
    def _enhance_details(self, result: Image.Image, original: Image.Image) -> Image.Image:
        """Enhance fine details in the result"""
        result_array = np.array(result)
        original_array = np.array(original)
        
        if result_array.shape[2] == 4:  # RGBA
            mask = result_array[:, :, 3]
            
            # Apply advanced post-processing
            refined_mask = self.processor.refine_edges(mask, original_array)
            enhanced_mask = self.processor.enhance_hair_details(refined_mask, original_array)
            
            # Update alpha channel
            result_array[:, :, 3] = enhanced_mask
        
        return Image.fromarray(result_array, 'RGBA')
    
    def _apply_edge_feathering(self, result: Image.Image) -> Image.Image:
        """Apply soft edge feathering"""
        result_array = np.array(result)
        
        if result_array.shape[2] == 4:  # RGBA
            mask = result_array[:, :, 3]
            feathered_mask = self.processor.apply_feathering(mask, feather_radius=5)
            result_array[:, :, 3] = feathered_mask
        
        return Image.fromarray(result_array, 'RGBA')

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='Professional Background Removal System')
    
    parser.add_argument('input', nargs='?', help='Input image path')
    parser.add_argument('-o', '--output', help='Output image path (optional)')
    parser.add_argument('-q', '--quality', choices=['good', 'high', 'ultra'], 
                       default='high', help='Processing quality level')
    parser.add_argument('-m', '--model', choices=list(ModelConfig.MODELS.keys()), 
                       help='Force specific model')
    parser.add_argument('--no-auto-detect', action='store_true', 
                       help='Disable automatic model selection')
    parser.add_argument('--no-enhance', action='store_true', 
                       help='Disable detail enhancement')
    parser.add_argument('--no-feather', action='store_true', 
                       help='Disable edge feathering')
    parser.add_argument('--list-models', action='store_true', 
                       help='List available models and exit')
    parser.add_argument('-v', '--verbose', action='store_true', 
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.list_models:
        print("\nAvailable Models:")
        print("=" * 50)
        for key, model in ModelConfig.MODELS.items():
            print(f"🎯 {key}")
            print(f"   Description: {model['description']}")
            print(f"   Best for: {', '.join(model['best_for'])}")
            print(f"   Quality: {model['quality']}")
            print()
        return
    
    # Validate input
    if not args.input:
        parser.print_help()
        return
        
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file '{args.input}' not found")
        return
    
    # Initialize professional system
    remover = ProfessionalBGRemover()
    
    try:
        start_time = datetime.now()
        
        # Process image
        output_path = remover.auto_remove_background(
            image_path=args.input,
            output_path=args.output,
            quality=args.quality,
            auto_detect=not args.no_auto_detect,
            model_override=args.model,
            enhance_details=not args.no_enhance,
            feather_edges=not args.no_feather
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ Professional background removal completed!")
        print(f"📁 Output: {output_path}")
        print(f"⏱️  Processing time: {processing_time:.2f}s")
        print(f"🎯 Quality: {args.quality.upper()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Processing failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
