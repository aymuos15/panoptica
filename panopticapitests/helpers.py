"""
Panoptica API Test Helpers
--------------------------
Utility functions for working with panoptic segmentation data, visualization,
and evaluation using the panopticapi format.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import ndimage
from matplotlib.patches import Patch
from PIL import Image
import json
import os
import tempfile
import copy
from pprint import pprint

###########################################################################
#                        VISUALIZATION FUNCTIONS                          #
###########################################################################

def plot_segmentation_comparison(pred, gt):
    """
    Plot a comparison of predicted and ground truth panoptic segmentations.
    
    Generates a 3-panel figure with:
    1. Prediction segmentation
    2. Ground truth segmentation
    3. Overlay of matches/mismatches between them
    
    Parameters:
        pred (numpy.ndarray): Predicted segmentation with class IDs
        gt (numpy.ndarray): Ground truth segmentation with class IDs
        
    Returns:
        dict: Summary of instance counts in prediction and ground truth
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Determine the number of classes dynamically
    num_classes = max(np.max(pred), np.max(gt))
    
    # Generate colors dynamically - use a colormap for larger number of classes
    if num_classes <= 5:
        # Use discrete colors for small number of classes
        base_colors = ["black", "blue", "red", "green", "purple", "orange", "cyan", "magenta", "yellow", "lime"]
        colors = base_colors[:int(num_classes) + 1]
    else:
        # Use a continuous colormap with distinct colors for many classes
        cmap = plt.cm.get_cmap('tab20', num_classes + 1)
        colors = ["black"] + [cmap(i) for i in range(num_classes)]
    
    # Create custom colormap for prediction and ground truth
    cmap_custom = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, num_classes + 1.5, 1), cmap_custom.N)
    
    # Plot with improved colormaps
    ax[0].imshow(pred, cmap=cmap_custom, norm=norm)
    ax[0].set_title("Prediction")
    
    ax[1].imshow(gt, cmap=cmap_custom, norm=norm)
    ax[1].set_title("Ground Truth")
    
    # For overlay visualization - create dynamic categories
    overlay = np.zeros_like(pred, dtype=np.int32)
    
    # Class matches - preserve the class ID
    for class_id in range(1, int(num_classes) + 1):
        match_mask = (pred == class_id) & (gt == class_id)
        overlay[match_mask] = class_id
    
    # Special categories for visualization
    mismatch = np.zeros_like(pred, dtype=np.int32)
    mismatch[(pred != gt) & (pred > 0) & (gt > 0)] = num_classes + 1  # Different classes
    
    only_pred = np.zeros_like(pred, dtype=np.int32)
    only_pred[(pred > 0) & (gt == 0)] = num_classes + 2  # Only in prediction
    
    only_gt = np.zeros_like(pred, dtype=np.int32)
    only_gt[(pred == 0) & (gt > 0)] = num_classes + 3  # Only in ground truth
    
    # Combine for visualization
    overlay_viz = overlay.copy()
    overlay_viz[mismatch > 0] = mismatch[mismatch > 0]
    overlay_viz[only_pred > 0] = only_pred[only_pred > 0]
    overlay_viz[only_gt > 0] = only_gt[only_gt > 0]
    
    # Custom colormap for overlay - with additional colors for special categories
    overlay_colors = colors.copy()  # Start with class colors
    # Add colors for special categories
    special_colors = ["yellow", "cyan", "magenta"]
    overlay_colors.extend(special_colors)
    
    overlay_cmap = mcolors.ListedColormap(overlay_colors)
    overlay_norm = mcolors.BoundaryNorm(np.arange(-0.5, num_classes + 4.5, 1), overlay_cmap.N)
    
    ax[2].imshow(overlay_viz, cmap=overlay_cmap, norm=overlay_norm)
    ax[2].set_title("Overlay")
    
    # Add legend for overlay
    legend_elements = [Patch(facecolor="black", label="Background")]
    
    # Add class match legends
    for class_id in range(1, int(num_classes) + 1):
        legend_elements.append(Patch(facecolor=colors[class_id], label=f"Class {class_id} Match"))
    
    # Add special category legends
    legend_elements.extend([
        Patch(facecolor="yellow", label="Class Mismatch"),
        Patch(facecolor="cyan", label="Only in Prediction"),
        Patch(facecolor="magenta", label="Only in Ground Truth"),
    ])
    
    ax[2].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left")
    
    # Process ground truth instances - Use a dictionary to store labeled instances by class
    gt_labeled = np.zeros_like(gt, dtype=np.int32)
    gt_instance_maps = {}  # Store instance maps by class
    gt_instance_count = {}  # Track instance count per class
    
    for class_id in range(1, int(num_classes) + 1):
        class_mask = gt == class_id
        if np.any(class_mask):
            # Label connected components for this class
            labeled, num_features = ndimage.label(class_mask)
            gt_instance_maps[class_id] = labeled.copy()
            gt_instance_count[class_id] = num_features
            
            # Store in the global labeled map (for visualization)
            for instance_id in range(1, num_features + 1):
                instance_mask = labeled == instance_id
                gt_labeled[instance_mask] = instance_id * 1000 + class_id  # Use a unique identifier
    
    # Process prediction instances - Use a dictionary to store labeled instances by class
    pred_labeled = np.zeros_like(pred, dtype=np.int32)
    pred_instance_maps = {}  # Store instance maps by class
    pred_instance_count = {}  # Track instance count per class
    
    for class_id in range(1, int(num_classes) + 1):
        class_mask = pred == class_id
        if np.any(class_mask):
            # Label connected components for this class
            labeled, num_features = ndimage.label(class_mask)
            pred_instance_maps[class_id] = labeled.copy()
            pred_instance_count[class_id] = num_features
            
            # Store in the global labeled map (for visualization)
            for instance_id in range(1, num_features + 1):
                instance_mask = labeled == instance_id
                pred_labeled[instance_mask] = instance_id * 1000 + class_id  # Use a unique identifier
    
    # Annotate instance numbers with better visibility
    # For ground truth
    for class_id in range(1, int(num_classes) + 1):
        labeled = gt_instance_maps.get(class_id, np.zeros_like(gt))
        for instance_id in range(1, np.max(labeled) + 1 if np.any(labeled) else 1):
            instance_mask = labeled == instance_id
            if np.any(instance_mask):
                y_coords, x_coords = np.where(instance_mask)
                y_center, x_center = int(np.mean(y_coords)), int(np.mean(x_coords))
                display_id = instance_id - 1  # Start from 0
                ax[1].text(
                    x_center,
                    y_center,
                    f"C{class_id}:{display_id}",
                    color="white",
                    fontsize=10,
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="black", alpha=0.5),
                )
    
    # For prediction
    for class_id in range(1, int(num_classes) + 1):
        labeled = pred_instance_maps.get(class_id, np.zeros_like(pred))
        for instance_id in range(1, np.max(labeled) + 1 if np.any(labeled) else 1):
            instance_mask = labeled == instance_id
            if np.any(instance_mask):
                y_coords, x_coords = np.where(instance_mask)
                y_center, x_center = int(np.mean(y_coords)), int(np.mean(x_coords))
                display_id = instance_id - 1  # Start from 0
                ax[0].text(
                    x_center,
                    y_center,
                    f"C{class_id}:{display_id}",
                    color="white",
                    fontsize=10,
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="black", alpha=0.5),
                )
    
    plt.tight_layout()
    plt.show()
    
    # Return instance counts for reporting
    return {
        "pred_instances": pred_instance_count,
        "gt_instances": gt_instance_count
    }

###########################################################################
#             PANOPTICAPI FORMAT CONVERSION FUNCTIONS                     #
###########################################################################

def create_panoptic_test_data(pred, gt, stuff_cls_list, thing_cls_list):
    """
    Convert numpy arrays to panoptic format required by panopticapi.
    
    Takes raw segmentation arrays and converts them to the format expected
    by the panopticapi evaluation tools, including creating the proper JSON
    structure and RGB-encoded image files.
    
    Parameters:
        pred (numpy.ndarray): Predicted segmentation
        gt (numpy.ndarray): Ground truth segmentation
        stuff_cls_list (list): List of class IDs for "stuff" categories
        thing_cls_list (list): List of class IDs for "thing" categories
        
    Returns:
        dict: Paths to the generated files and directories
    """
    # Create temporary directory to store images
    temp_dir = tempfile.mkdtemp()
    gt_folder = os.path.join(temp_dir, 'gt')
    pred_folder = os.path.join(temp_dir, 'pred')
    os.makedirs(gt_folder, exist_ok=True)
    os.makedirs(pred_folder, exist_ok=True)
    
    # Create category definitions
    categories = []
    
    # Add stuff classes only if the list is provided
    if stuff_cls_list:
        for cls_id in stuff_cls_list:
            categories.append({
                "id": int(cls_id),
                "name": f"stuff_{cls_id}",
                "supercategory": "stuff",
                "isthing": 0,  # This is correct - stuff should be 0
                "color": [100, 100, 100]
            })
    
    # Add thing classes only if the list is provided
    if thing_cls_list:
        for cls_id in thing_cls_list:
            categories.append({
                "id": int(cls_id),
                "name": f"thing_{cls_id}",
                "supercategory": "thing",
                "isthing": 1,  # This is correct - thing should be 1
                "color": [200, 0, 0]
            })
    
    # Setup JSON structure
    gt_json = {
        "images": [{"id": 1, "file_name": "dummy_gt.png", "height": gt.shape[0], "width": gt.shape[1]}],
        "annotations": [],
        "categories": categories
    }
    
    pred_json = {
        "images": [{"id": 1, "file_name": "dummy_pred.png", "height": pred.shape[0], "width": pred.shape[1]}],
        "annotations": [],
        "categories": categories
    }
    
    # Convert gt to RGB panoptic format and extract segments_info
    gt_rgb, gt_segments = convert_to_panoptic_format(gt, 1, stuff_cls_list, thing_cls_list)
    pred_rgb, pred_segments = convert_to_panoptic_format(pred, 1, stuff_cls_list, thing_cls_list)
    
    # Save images
    gt_path = os.path.join(gt_folder, "dummy_gt.png")
    pred_path = os.path.join(pred_folder, "dummy_pred.png")
    
    Image.fromarray(gt_rgb).save(gt_path)
    Image.fromarray(pred_rgb).save(pred_path)
    
    # Add annotations
    gt_json["annotations"].append({
        "image_id": 1,
        "file_name": "dummy_gt.png",
        "segments_info": gt_segments
    })
    
    pred_json["annotations"].append({
        "image_id": 1,
        "file_name": "dummy_pred.png",
        "segments_info": pred_segments
    })
    
    # Save JSONs
    gt_json_path = os.path.join(temp_dir, "gt.json")
    pred_json_path = os.path.join(temp_dir, "pred.json")
    
    with open(gt_json_path, 'w') as f:
        json.dump(gt_json, f)
    
    with open(pred_json_path, 'w') as f:
        json.dump(pred_json, f)
    
    return {
        "gt_json": gt_json_path,
        "pred_json": pred_json_path,
        "gt_folder": gt_folder,
        "pred_folder": pred_folder,
        "temp_dir": temp_dir
    }

def convert_to_panoptic_format(segmentation, image_id, stuff_cls_list, thing_cls_list):
    """
    Convert a segmentation array to panoptic format with unique IDs using one-hot encoding.
    """
    height, width = segmentation.shape
    rgb_segmentation = np.zeros((height, width, 3), dtype=np.uint8)
    segments_info = []
    
    # Start with ID 1 (0 is typically background)
    segment_id = 1
    
    # Process stuff classes only if the list is provided and not empty
    if stuff_cls_list:
        for cls_id in stuff_cls_list:
            # Create a mask for this stuff class
            mask = (segmentation == cls_id)
            if not np.any(mask):
                continue
                
            area = np.sum(mask)
            
            # Get the bounding box
            y_indices, x_indices = np.where(mask)
            if len(y_indices) == 0:
                continue
                
            bbox = [
                int(np.min(x_indices)),
                int(np.min(y_indices)),
                int(np.max(x_indices) - np.min(x_indices) + 1),
                int(np.max(y_indices) - np.min(y_indices) + 1)
            ]
            
            # Create a unique color for this segment
            r = segment_id % 256
            g = (segment_id // 256) % 256
            b = (segment_id // (256**2)) % 256
            
            # Assign the color to the segmentation
            rgb_segmentation[mask] = [r, g, b]
            
            # Add segment info
            segments_info.append({
                "id": int(segment_id),
                "category_id": int(cls_id),
                "iscrowd": 0,
                "bbox": bbox,
                "area": int(area)
            })
            
            segment_id += 1
    
    # Process thing classes with connected components
    if thing_cls_list: # Also good practice to check thing_cls_list
        for cls_id in thing_cls_list:
            # Create a one-hot mask for this thing class
            mask = (segmentation == cls_id)
            if not np.any(mask):
                continue
                
            # Find connected components (instances) for this thing class
            from scipy.ndimage import label
            labeled_mask, num_components = label(mask)
            
            for i in range(1, num_components + 1):
                component_mask = (labeled_mask == i)
                area = np.sum(component_mask)
                
                if area < 1:  # Skip tiny regions
                    continue
                    
                # Get the bounding box
                y_indices, x_indices = np.where(component_mask)
                bbox = [
                    int(np.min(x_indices)),
                    int(np.min(y_indices)),
                    int(np.max(x_indices) - np.min(x_indices) + 1),
                    int(np.max(y_indices) - np.min(y_indices) + 1)
                ]
                
                # Create a unique color for this segment
                r = segment_id % 256
                g = (segment_id // 256) % 256
                b = (segment_id // (256**2)) % 256
                
                # Assign the color to the segmentation
                rgb_segmentation[component_mask] = [r, g, b]
                
                # Add segment info with the original class ID
                segments_info.append({
                    "id": int(segment_id),
                    "category_id": int(cls_id),
                    "iscrowd": 0,
                    "bbox": bbox,
                    "area": int(area)
                })
                
                segment_id += 1
    
    return rgb_segmentation, segments_info

###########################################################################
#                        JSON HANDLING FUNCTIONS                          #
###########################################################################

def simplify_coco_json(json_data):
    """
    Create a simplified version of a COCO-format JSON for better readability.
    
    Takes a full COCO or panoptic JSON and produces a compact version
    with fewer examples of each section to make it easier to understand.
    
    Parameters:
        json_data (dict): The original COCO or panoptic format JSON data
        
    Returns:
        dict: Simplified version of the JSON data with fewer examples
    """
    # Create a deep copy to avoid modifying the original
    simplified = copy.deepcopy(json_data)
    
    # Keep only first license
    if "licenses" in simplified and len(simplified["licenses"]) > 1:
        simplified["licenses"] = [simplified["licenses"][0], "..."]
    
    # Keep only first image
    if "images" in simplified and len(simplified["images"]) > 1:
        simplified["images"] = [simplified["images"][0], "..."]
    
    # Keep only first annotation but show structure
    if "annotations" in simplified and len(simplified["annotations"]) > 1:
        # Keep first annotation but limit segments_info
        if simplified["annotations"][0].get("segments_info") and len(simplified["annotations"][0]["segments_info"]) > 3:
            simplified["annotations"][0]["segments_info"] = simplified["annotations"][0]["segments_info"][:3] + ["..."]
        simplified["annotations"] = [simplified["annotations"][0], "..."]
    
    # Keep only a few categories to show structure
    if "categories" in simplified and len(simplified["categories"]) > 5:
        # Keep first few categories of different types
        # Find examples of "thing" and "stuff" categories
        things = [cat for cat in simplified["categories"][:10] if cat.get("isthing") == 1][:3]
        stuff = [cat for cat in simplified["categories"][:20] if cat.get("isthing") == 0][:2]
        simplified["categories"] = things + stuff + ["..."]
    
    return simplified

def print_formatted_json(json_data, indent=2):
    """
    Print JSON data with nice formatting.
    
    Parameters:
        json_data (dict): The JSON data to print
        indent (int): Indentation level for formatting
    """
    formatted_json = json.dumps(json_data, indent=indent)
    print(formatted_json)

def load_and_print_json(filename):
    """
    Load JSON from file and print it with formatting.
    
    Parameters:
        filename (str): Path to the JSON file
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    print_formatted_json(data)
    
def parse_and_print_json(json_string):
    """
    Parse JSON string and print it with formatting.
    
    Parameters:
        json_string (str): JSON content as a string
    """
    data = json.loads(json_string)
    print_formatted_json(data)

# Example usage:
# load_and_print_json('./sample_data/panoptic_examples.json')