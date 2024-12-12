import SimpleITK as sitk
import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt

class NiftiImageViewer:
    def __init__(self, image_raw: sitk.Image, image_pp: sitk.Image, image_gt: sitk.Image, filepath1: str, filepath2: str):
        # Load images
        self.image_raw = image_raw
        self.image_pp = image_pp
        self.image_gt = image_gt
        self.axis = 0  # Axial
        self.axis_labels = ["Axial", "Coronal", "Sagittal"]

        # Ensure orientation matches the raw image
        self.normalize_orientation()

        # Metadata
        self.size = self.image_raw.GetSize()
        self.spacing = self.image_raw.GetSpacing()
        self.orientation = self.image_raw.GetDirection()
        self.origin = self.image_raw.GetOrigin()

        # Initialize slice index (middle slice by default)
        self.current_slice = self.size[0] // 2

        # Extract PatientID from the file paths
        self.patient_id_raw = self.extract_patient_id(filepath1)
        self.patient_id_pp = self.extract_patient_id(filepath2)

        # Load all DICE and HDRFDST values for raw and processed images
        self.dice_values_raw, self.hdrfdst_values_raw = self.get_metrics(filepath1)
        self.dice_values_pp, self.hdrfdst_values_pp = self.get_metrics(filepath2)

    def normalize_orientation(self):
        """Align orientation, origin, and spacing of GT and PP images to match the raw image."""
        reference_origin = self.image_raw.GetOrigin()
        reference_spacing = self.image_raw.GetSpacing()
        reference_direction = self.image_raw.GetDirection()

        # Align ground truth image
        self.image_gt = self.adjust_image_geometry(self.image_gt, reference_origin, reference_spacing, reference_direction)

        # Align post-processed image
        self.image_pp = self.adjust_image_geometry(self.image_pp, reference_origin, reference_spacing, reference_direction)

    def adjust_image_geometry(self, image, reference_origin, reference_spacing, reference_direction):
        """Adjust the origin, spacing, and direction of an image to match the reference."""
        aligned_image = sitk.Image(image)  # Create a copy
        aligned_image.SetOrigin(reference_origin)
        aligned_image.SetSpacing(reference_spacing)
        aligned_image.SetDirection(reference_direction)
        return aligned_image

    def extract_patient_id(self, filepath: str) -> str:
        """Extract the PatientID from the file name."""
        filename = os.path.basename(filepath)
        match = re.match(r"(\d+)_SEG(?:-(\w+))?\.mha", filename)
        if not match:
            raise ValueError(f"Invalid filename format: {filename}. Could not extract PatientID.")
        patient_id = match.group(1)
        suffix = match.group(2)  # Capture suffix after "-".
        return f"{patient_id}-{suffix}" if suffix else patient_id

    def get_metrics(self, filepath: str) -> tuple:
        """Extract all DICE and HDRFDST values for the given PatientID and their corresponding labels."""
        results_path = os.path.dirname(filepath)
        csv_path = os.path.join(results_path, "results.csv")
        patient_id = self.extract_patient_id(filepath)
        try:
            # Load the CSV file
            df = pd.read_csv(csv_path, delimiter=';')

            # Filter rows for the given PatientID
            patient_rows = df[df['SUBJECT'] == patient_id]
            if patient_rows.empty:
                raise ValueError(f"No metrics found for Patient ID: {patient_id} in {csv_path}")

            # Extract DICE and HDRFDST values into dictionaries
            dice_values = dict(zip(patient_rows['LABEL'], patient_rows['DICE']))
            hdrfdst_values = dict(zip(patient_rows['LABEL'], patient_rows['HDRFDST']))
            return dice_values, hdrfdst_values
        except FileNotFoundError:
            raise FileNotFoundError(f"The results file was not found at {csv_path}")
        except KeyError as e:
            raise KeyError(f"Expected column '{e.args[0]}' is missing in the results file.")

    def display(self):
        """Interactive display of the three images."""
        raw_dice_str = ", ".join([f"{label} ({value:.2f})" for label, value in self.dice_values_raw.items()])
        pp_dice_str = ", ".join([f"{label} ({value:.2f})" for label, value in self.dice_values_pp.items()])
        raw_hdrfdst_str = ", ".join([f"{label} ({value:.2f})" for label, value in self.hdrfdst_values_raw.items()])
        pp_hdrfdst_str = ", ".join([f"{label} ({value:.2f})" for label, value in self.hdrfdst_values_pp.items()])

        # Set up the figure and axes
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(15, 5))
        self.ax1.set_title(f"Ground Truth Labels", fontsize=14)
        self.ax2.set_title(f"Raw Segmentation", fontsize=14)
        self.ax3.set_title(f"PP Segmentation", fontsize=14)
        self.ax1.axis('off')
        self.ax2.axis('off')
        self.ax3.axis('off')

        # Display initial slices
        self.update_display(initial=True)

        # Add navigation instructions
        self.fig.suptitle(
            f"CURRENT VIEW: {self.axis_labels[self.axis]} - Slice {self.current_slice}\n"
            f"\nRAW DICES: {raw_dice_str}\nPP DICES: {pp_dice_str}\n"
            f"\nRAW HDRFDST: {raw_hdrfdst_str}\nPP HDRFDST: {pp_hdrfdst_str}",
            fontsize=20
        )

        # Connect events for scrolling and axis switching
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_keypress)

        # Show the figure
        plt.tight_layout(rect=[0, 0, 1, 1.2])
        plt.show()

    def on_scroll(self, event):
        """Handle mouse wheel events for scrolling through slices."""
        if event.button == 'up':  # Scroll up
            self.current_slice = min(self.current_slice + 1, self.size[self.axis] - 1)
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
        self.current_slice = self.size[self.axis] // 2
        self.update_display()

    def update_display(self, initial=False):
        """Update the displayed slices for the current axis and slice."""
        if self.axis == 0:
            slice1 = np.fliplr(sitk.GetArrayViewFromImage(self.image_gt)[self.current_slice, :, :])
            slice2 = sitk.GetArrayViewFromImage(self.image_raw)[self.current_slice, :, :]
            slice3 = sitk.GetArrayViewFromImage(self.image_pp)[self.current_slice, :, :]
        elif self.axis == 1:
            slice1 = sitk.GetArrayViewFromImage(self.image_gt)[:, self.current_slice, :]
            slice2 = sitk.GetArrayViewFromImage(self.image_raw)[:, self.current_slice, :]
            slice3 = sitk.GetArrayViewFromImage(self.image_pp)[:, self.current_slice, :]
        elif self.axis == 2:
            slice1 = sitk.GetArrayViewFromImage(self.image_gt)[:, :, self.current_slice]
            slice2 = sitk.GetArrayViewFromImage(self.image_raw)[:, :, self.current_slice]
            slice3 = sitk.GetArrayViewFromImage(self.image_pp)[:, :, self.current_slice]

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

        # Update subplot titles
        self.ax1.set_title(f"Ground Truth Labels", fontsize=20)
        self.ax2.set_title(f"Raw Segmentation", fontsize=20)
        self.ax3.set_title(f"PP Segmentation", fontsize=20)

        # Update the figure title with DICE and HDRFDST values
        raw_dice_str = ", ".join([f"{label} ({value:.2f})" for label, value in self.dice_values_raw.items()])
        pp_dice_str = ", ".join([f"{label} ({value:.2f})" for label, value in self.dice_values_pp.items()])
        raw_hdrfdst_str = ", ".join([f"{label} ({value:.2f})" for label, value in self.hdrfdst_values_raw.items()])
        pp_hdrfdst_str = ", ".join([f"{label} ({value:.2f})" for label, value in self.hdrfdst_values_pp.items()])
        self.fig.suptitle(
            f"CURRENT VIEW: {self.axis_labels[self.axis]} - Slice {self.current_slice}\n"
            f"\nRAW DICES: {raw_dice_str}\nPP DICES: {pp_dice_str}\n"
            f"\nRAW HDRFDST: {raw_hdrfdst_str}\nPP HDRFDST: {pp_hdrfdst_str}",
            fontsize=20
        )

        # Refresh the canvas
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    filepath_gt = "dataset/test/118528/labels_native.nii.gz"  # Ground truth labels
    filepath_raw = "mia-result/2024-11-18-19-46-59 (noPP, ne20_md50)/118528_SEG.mha"  # Raw segmentation
    #filepath_pp = "mia-result/2024-12-09_ddOpening&2DHF/2024-12-07-17-08-58 (ddO&2DHF50)/validation/117122_SEG-PP.mha"  # Post-processed segmentation (validation)
    filepath_pp = "mia-result/2024-12-09_ddOpening&2DHF/2024-12-07-17-08-58 (ddO&2DHF50)/testing/118528_SEG-PP.mha"  # Post-processed segmentation (testing)

    gt = sitk.ReadImage(filepath_gt)
    raw = sitk.ReadImage(filepath_raw)
    pp = sitk.ReadImage(filepath_pp)

    print("Ground truth size:", gt.GetSize())
    print("Raw image size:", raw.GetSize())
    print("PP image size:", pp.GetSize())

    viewer = NiftiImageViewer(raw, pp, gt, filepath_raw, filepath_pp)
    viewer.display()
