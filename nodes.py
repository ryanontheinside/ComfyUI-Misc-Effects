import torch
import numpy as np

class StarburstNode:
    """Node that creates a starburst effect from the center of a mask"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "num_rays": ("INT", {"default": 60, "min": 4, "max": 360, "step": 1}),
                "ray_thickness": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "fill": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "create_starburst"
    CATEGORY = "mask/effect"

    def create_starburst(self, mask, num_rays, ray_thickness, fill):
        batch_size = mask.shape[0]
        device = mask.device
        
        # Convert entire batch to numpy at once
        masks_np = mask.squeeze().cpu().numpy()
        if len(masks_np.shape) == 2:  # Single mask case
            masks_np = masks_np[None, ...]
        
        # Process each mask in the batch
        results = []
        for b in range(batch_size):
            mask_np = masks_np[b]
            
            #center calculation
            y_coords, x_coords = np.where(mask_np > 0)
            if len(y_coords) == 0:
                # Return empty starburst if mask is empty
                height, width = mask_np.shape
                results.append(np.zeros((height, width), dtype=np.float32))
                continue
            
            center_y = int(y_coords.mean())
            center_x = int(x_coords.mean())
            
            # Create output array directly
            height, width = mask_np.shape
            starburst = np.zeros((height, width), dtype=np.float32)
            
            if fill:
                y, x = np.mgrid[:height, :width]
                y = y - center_y
                x = x - center_x
                pixel_angles = np.arctan2(y, x)
                pixel_angles = np.where(pixel_angles < 0, pixel_angles + 2*np.pi, pixel_angles)
                
                # Calculate ray angles once
                ray_angles = np.linspace(0, 2*np.pi, num_rays, endpoint=False)
                
                # Create the filled starburst in one operation
                for i in range(0, num_rays, 2):  # Step by 2 to fill every other section
                    current_angle = ray_angles[i]
                    next_angle = ray_angles[(i + 1) % num_rays]
                    if next_angle < current_angle:
                        next_angle += 2*np.pi
                    mask = (pixel_angles >= current_angle) & (pixel_angles <= next_angle)
                    starburst[mask] = 1.0
            else:
                # Pre-calculate angles and endpoints for rays
                angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
                max_radius = np.sqrt(width**2 + height**2)
                
                # Calculate all endpoints at once
                end_x = center_x + max_radius * np.cos(angles)
                end_y = center_y + max_radius * np.sin(angles)
                
                # Create coordinate grids once
                y, x = np.ogrid[:height, :width]
                
                # Vectorize ray drawing
                for i in range(num_rays):
                    dx = end_x[i] - center_x
                    dy = end_y[i] - center_y
                    length = np.sqrt(dx*dx + dy*dy)
                    dx, dy = dx/length, dy/length
                    dist = np.abs((x - center_x) * (-dy) + (y - center_y) * dx)
                    starburst[dist < ray_thickness/2] = 1.0
            
            results.append(starburst)
        
        starburst_tensor = torch.from_numpy(np.stack(results)).float()
        return (starburst_tensor.to(device),)

NODE_CLASS_MAPPINGS = {
    "StarburstNode": StarburstNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarburstNode": "Starburst"
}