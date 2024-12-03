import SimpleITK as sitk
import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt

def resample_image_to_match(image, reference_image):
    """Resample an image to match the reference image's size, spacing, and orientation."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Use nearest neighbor for label images
    resampler.SetDefaultPixelValue(0)  # Background value
    return resampler.Execute(image)

class NiftiImageViewer:
    def __init__(self, image_raw: sitk.Image, image_pp: sitk.Image, image_gt: sitk.Image, filepath1: str, filepath2: str):
        # Load images and their arrays
        self.image_raw = image_raw
        self.image_pp = image_pp
        self.image_gt = image_gt
        self.image_array1 = sitk.GetArrayFromImage(self.image_raw)
        self.image_array2 = sitk.GetArrayFromImage(self.image_pp)
        self.image_array3 = sitk.GetArrayFromImage(self.image_gt)
        self.axis = 0  # Axial
        self.axis_labels = ["Axial", "Coronal", "Sagittal"]

        # Verify all images have the same dimensions
        if not (self.image_array1.shape == self.image_array2.shape == self.image_array3.shape):
            raise ValueError("All images (raw, processed, ground truth) must have the same dimensions!")

        # Metadata
        self.size = self.image_raw.GetSize()
        self.spacing = self.image_raw.GetSpacing()
        self.orientation = self.image_raw.GetDirection()
        self.origin = self.image_raw.GetOrigin()

        # Initialize slice index (middle slice by default)
        self.current_slice = self.image_array1.shape[0] // 2

        # Extract PatientID from the file paths
        self.patient_id_raw = self.extract_patient_id(filepath1)
        self.patient_id_pp = self.extract_patient_id(filepath2)

        # Load all DICE values for raw and processed images
        self.dice_values_raw = self.get_dice_values(filepath1)
        self.dice_values_pp = self.get_dice_values(filepath2)

    def extract_patient_id(self, filepath: str) -> str:
        """Extract the PatientID from the file name."""
        filename = os.path.basename(filepath)
        match = re.match(r"(\d+)_SEG(?:-(\w+))?\.mha", filename)
        if not match:
            raise ValueError(f"Invalid filename format: {filename}. Could not extract PatientID.")
        patient_id = match.group(1)
        suffix = match.group(2)  # This will capture "PP" or any other suffix after "-".
        return f"{patient_id}-{suffix}" if suffix else patient_id

    def get_dice_values(self, filepath: str) -> dict:
        """Extract all DICE values for the given PatientID and their corresponding labels."""
        results_path = os.path.dirname(filepath)
        csv_path = os.path.join(results_path, "results.csv")
        patient_id = self.extract_patient_id(filepath)
        try:
            # Load the CSV file
            df = pd.read_csv(csv_path, delimiter=';')

            # Filter rows for the given PatientID
            patient_rows = df[df['SUBJECT'] == patient_id]
            if patient_rows.empty:
                raise ValueError(f"No DICE values found for Patient ID: {patient_id} in {csv_path}")

            # Extract labels and DICE values into a dictionary
            dice_values = dict(zip(patient_rows['LABEL'], patient_rows['DICE']))
            return dice_values
        except FileNotFoundError:
            raise FileNotFoundError(f"The results file was not found at {csv_path}")
        except KeyError as e:
            raise KeyError(f"Expected column '{e.args[0]}' is missing in the results file.")

    def display(self):
        """Interactive display of the three images."""
        raw_dice_str = ", ".join([f"{label} ({value:.3f})" for label, value in self.dice_values_raw.items()])
        pp_dice_str = ", ".join([f"{label} ({value:.3f})" for label, value in self.dice_values_pp.items()])

        # Set up the figure and axes
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(15, 5))
        self.ax1.set_title(f"Ground Truth - Slice {self.current_slice}")
        self.ax2.set_title(f"Raw Segmentation - Slice {self.current_slice}")
        self.ax3.set_title(f"PP Segmentation - Slice {self.current_slice}")
        self.ax1.axis('off')
        self.ax2.axis('off')
        self.ax3.axis('off')

        # Display initial slices
        self.update_display(initial=True)

        # Add navigation instructions
        self.fig.suptitle(
            f"CURRENT AXIS: {self.axis_labels[self.axis]}\n"
            f"RAW DICES: {raw_dice_str}\n"
            f"PP DICES: {pp_dice_str}",
            fontsize=20  # Increased font size for the main title
        )

        # Connect events for scrolling and axis switching
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
        """Handle key press events for switching axes."""
        if event.key == 'right':  # Move to the next axis
            self.axis = (self.axis + 1) % 3
        elif event.key == 'left':  # Move to the previous axis
            self.axis = (self.axis - 1) % 3
        else:
            return  # Ignore other keys

        print(f"Changing axis to {self.axis} ({self.axis_labels[self.axis]})")
        # Update the current slice to the middle slice of the new axis
        self.current_slice = self.image_array1.shape[self.axis] // 2
        self.update_display()

    def update_display(self, initial=False):
        """Update the displayed slices for the current axis and slice."""
        if self.axis == 0:
            slice1 = self.image_array3[self.current_slice, :, :]
            slice2 = self.image_array1[self.current_slice, :, :]
            slice3 = self.image_array2[self.current_slice, :, :]
        elif self.axis == 1:
            slice1 = self.image_array3[:, self.current_slice, :]
            slice2 = self.image_array1[:, self.current_slice, :]
            slice3 = self.image_array2[:, self.current_slice, :]
        elif self.axis == 2:
            slice1 = self.image_array3[:, :, self.current_slice]
            slice2 = self.image_array1[:, :, self.current_slice]
            slice3 = self.image_array2[:, :, self.current_slice]

        if initial:
            # Initialize the image displays
            self.im_display1 = self.ax1.imshow(slice1, cmap='viridis', origin='lower', vmin=0, vmax=5)
            self.im_display2 = self.ax2.imshow(slice2, cmap='viridis', origin='lower', vmin=0, vmax=5)
            self.im_display3 = self.ax3.imshow(slice3, cmap='viridis', origin='lower', vmin=0, vmax=5)
        else:
            # Update the data for existing image displays
            self.im_display1.set_data(slice1)
            self.im_display2.set_data(slice2)
            self.im_display3.set_data(slice3)

        # Update the figure title to reflect the current axis and slice
        raw_dice_str = ", ".join([f"{label} ({value:.3f})" for label, value in self.dice_values_raw.items()])
        pp_dice_str = ", ".join([f"{label} ({value:.3f})" for label, value in self.dice_values_pp.items()])
        self.fig.suptitle(
            f"CURRENT AXIS: {self.axis_labels[self.axis]}\n"
            f"RAW DICES: {raw_dice_str}\n"
            f"PP DICES: {pp_dice_str}",
            fontsize=20  # Increased font size for the main title
        )

        # Update subplot titles with a larger font size
        self.ax1.set_title(f"Ground Truth - Slice {self.current_slice}", fontsize=20)
        self.ax2.set_title(f"Raw Segmentation - Slice {self.current_slice}", fontsize=20)
        self.ax3.set_title(f"PP Segmentation - Slice {self.current_slice}", fontsize=20)

        # Refresh the canvas
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    filepath_gt = "dataset/test/117122/labels_native.nii.gz"  # Ground truth labels
    filepath_raw = "mia-result/2024-11-18-19-46-59 (noPP, ne20_md50)/117122_SEG.mha"  # Raw segmentation
    filepath_pp = "mia-result/2024-11-27-20-44-34 (PP-ddO, ne20_md50)/117122_SEG-PP.mha"  # Post-processed segmentation

    gt = sitk.ReadImage(filepath_gt)
    raw = sitk.ReadImage(filepath_raw)
    pp = sitk.ReadImage(filepath_pp)

    print("Ground truth size:", gt.GetSize())
    print("Raw image size:", raw.GetSize())
    print("Processed image size:", pp.GetSize())

    # Resample ground truth to match raw segmentation
    gt_resampled = resample_image_to_match(gt, raw)

    print("Resampled ground truth size:", gt_resampled.GetSize())

    viewer = NiftiImageViewer(raw, pp, gt_resampled, filepath_raw, filepath_pp)
    viewer.display()
