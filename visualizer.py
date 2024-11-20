import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

class NiftiImageViewer:
    def __init__(self, image_raw:sitk.Image, image_pp:sitk.Image):
        # Load the NIFTI images
        self.image_raw = image_raw
        self.image_pp = image_pp
        self.image_array1 = sitk.GetArrayFromImage(self.image_raw)
        self.image_array2 = sitk.GetArrayFromImage(self.image_pp)
        self.axis = 0 # coronal, axial, saggital
        
        # Verify both images have the same dimensions
        if self.image_array1.shape != self.image_array2.shape:
            raise ValueError("Both images must have the same dimensions!")
        
        # Metadata
        self.size = self.image_raw.GetSize()
        self.spacing = self.image_raw.GetSpacing()
        self.orientation = self.image_raw.GetDirection()
        self.origin = self.image_raw.GetOrigin()

        # Initialize slice index (middle slice by default)
        self.current_slice = self.image_array1.shape[0] // 2
        
        print(f"Image 1 and Image 2 loaded with dimensions: {self.image_array1.shape}")
        print(f"Size: {self.size}, Spacing: {self.spacing}, Origin: {self.origin}")
        print(f"Orientation matrix: {self.orientation}")
        print("Use mousewheel to scroll through slices. Press 'space' and then scroll to change the axis (coronal, axial, saggital). You can only cycle through them.")
    
    def display(self):
        """Interactive display of the two images."""
        # Set up the figure and axes
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self.ax1.set_title(f"Image 1 - Slice {self.current_slice}")
        self.ax2.set_title(f"Image 2 - Slice {self.current_slice}")
        self.ax1.axis('off')
        self.ax2.axis('off')

        # Display the initial slices
        if self.axis == 0:
            disp_slice1 = self.image_array1[self.current_slice, :, :]
            disp_slice2 = self.image_array2[self.current_slice, :, :]
        if self.axis == 1:
            disp_slice1 = self.image_array1[:,self.current_slice,:]
            disp_slice2 = self.image_array2[:,self.current_slice,:]
        if self.axis == 2:
            disp_slice1 = self.image_array1[:,:,self.current_slice]
            disp_slice2 = self.image_array2[:,:,self.current_slice]

        self.im_display1 = self.ax1.imshow(
            disp_slice1,
            cmap='viridis',
            origin='lower',
            vmin=0,
            vmax=5
        )
        self.im_display2 = self.ax2.imshow(
            disp_slice2,
            cmap='viridis',
            origin='lower',
            vmin=0,
            vmax=5
        )
        
        # Connect the mouse wheel event for scrolling
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_keypress)
        
        # Show the figure
        plt.tight_layout()
        plt.show()

    def on_scroll(self, event):
        """Handle mouse wheel events for scrolling through slices."""
        if event.button == 'up':  # Scroll up
            self.current_slice = min(self.current_slice + 1, self.image_array1.shape[self.axis] - 1)
        elif event.button == 'down':  # Scroll down
            self.current_slice = max(self.current_slice - 1, 0)

        # Update the displayed slices
        self.update_display()

    def on_keypress(self, event):
        if event.key == ' ':
            print("Changing axis")
            self.axis += 1
            if self.axis % 3 == 0:
                self.axis = 0
            self.current_slice = self.image_array1.shape[self.axis] // 2

    def update_display(self):
        """Update the image display for the current slice."""
        if self.axis == 0:
            self.im_display1.set_data(self.image_array1[self.current_slice, :, :])
            self.im_display2.set_data(self.image_array2[self.current_slice, :, :])
        if self.axis == 1:
            self.im_display1.set_data(self.image_array1[:, self.current_slice, :])
            self.im_display2.set_data(self.image_array2[:, self.current_slice, :])
        if self.axis == 2:
            self.im_display1.set_data(self.image_array1[:,:,self.current_slice])
            self.im_display2.set_data(self.image_array2[:,:,self.current_slice])
        self.ax1.set_title(f"Image 1 - Slice {self.current_slice}")
        self.ax2.set_title(f"Image 2 - Slice {self.current_slice}")
        self.fig.canvas.draw_idle()


# Example usage
if __name__ == "__main__":
    from mialab.filtering.postprocessing import ImagePostProcessing
    filepath1 = "mia-result/2024-11-19-17-56-35/117122_SEG.mha"  # Replace with the first image file path
    filepath2 = "mia-result/2024-11-19-17-56-35/117122_SEG.mha"  # Replace with the second image file path

    raw = sitk.ReadImage(filepath1)
    pp = sitk.ReadImage(filepath2)

    # Add evaluation.
    # Maybe add other views.
    processor = ImagePostProcessing()
    pp = processor.execute(pp)

    viewer = NiftiImageViewer(raw, pp)
    viewer.display()
