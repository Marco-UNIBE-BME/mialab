import SimpleITK as sitk
import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt

class NiftiImageViewer:
    def __init__(self, image_raw: sitk.Image, image_pp: sitk.Image, filepath1: str, filepath2: str):
        # Load the NIFTI images
        self.image_raw = image_raw
        self.image_pp = image_pp
        self.image_array1 = sitk.GetArrayFromImage(self.image_raw)
        self.image_array2 = sitk.GetArrayFromImage(self.image_pp)
        self.axis = 0  # Axial
        self.axis_labels = ["Axial", "Coronal", "Sagittal"]

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

        # Extract PatientID from the file paths
        self.patient_id_raw = self.extract_patient_id(filepath1)
        self.patient_id_pp = self.extract_patient_id(filepath2)

        # Load all DICE values for both raw and processed images
        self.dice_values_raw = self.get_dice_values(filepath1)
        self.dice_values_pp = self.get_dice_values(filepath2)

        print(f"Patient {self.patient_id_raw} DICE Values (Raw): {self.dice_values_raw}")
        print(f"Patient {self.patient_id_pp} DICE Values (PP): {self.dice_values_pp}")

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
        """Interactive display of the two images."""
        # Concatenate DICE values and labels into a readable string
        raw_dice_str = ", ".join([f"{label} ({value:.3f})" for label, value in self.dice_values_raw.items()])
        pp_dice_str = ", ".join([f"{label} ({value:.3f})" for label, value in self.dice_values_pp.items()])


# Set up the figure and axes
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self.ax1.set_title(f"Raw Segmentation - Slice {self.current_slice}")
        self.ax2.set_title(f"PP Segmentation - Slice {self.current_slice}")
        self.ax1.axis('off')
        self.ax2.axis('off')

        # Set the initial axis and DICE value display
        self.fig.suptitle(
            f"CURRENT AXIS: {self.axis_labels[self.axis]}\n"
            f"RAW DICES: {raw_dice_str}\n"
            f"PP DICES: {pp_dice_str}"
        )

        # Display the initial slices
        if self.axis == 0:
            disp_slice1 = self.image_array1[self.current_slice, :, :]
            disp_slice2 = self.image_array2[self.current_slice, :, :]
        if self.axis == 1:
            disp_slice1 = self.image_array1[:, self.current_slice, :]
            disp_slice2 = self.image_array2[:, self.current_slice, :]
        if self.axis == 2:
            disp_slice1 = self.image_array1[:, :, self.current_slice]
            disp_slice2 = self.image_array2[:, :, self.current_slice]

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

        # Connect the mouse wheel event for scrolling and keyboard for axis switching
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
        self.current_slice = self.image_array1.shape[self.axis] // 2
        self.update_display()

    def update_display(self):
        """Update the image display for the current slice."""
        if self.axis == 0:
            self.im_display1.set_data(self.image_array1[self.current_slice, :, :])
            self.im_display2.set_data(self.image_array2[self.current_slice, :, :])
        if self.axis == 1:
            self.im_display1.set_data(self.image_array1[:, self.current_slice, :])
            self.im_display2.set_data(self.image_array2[:, self.current_slice, :])
        if self.axis == 2:
            self.im_display1.set_data(self.image_array1[:, :, self.current_slice])
            self.im_display2.set_data(self.image_array2[:, :, self.current_slice])

        # Concatenate DICE values and labels into a readable string
        raw_dice_str = ", ".join([f"{label} ({value:.3f})" for label, value in self.dice_values_raw.items()])
        pp_dice_str = ", ".join([f"{label} ({value:.3f})" for label, value in self.dice_values_pp.items()])


# Update the figure title to show the current axis and DICE values
        self.fig.suptitle(
            f"CURRENT AXIS: {self.axis_labels[self.axis]}\n"
            f"RAW DICES: {raw_dice_str}\n"
            f"PP DICES: {pp_dice_str}"
        )
        self.ax1.set_title(f"Raw Segmentation - Slice {self.current_slice}")
        self.ax2.set_title(f"PP Segmentation - Slice {self.current_slice}")
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    filepath1 = "mia-result/2024-11-18-19-46-59 (noPP, ne20_md50)/117122_SEG.mha"  # Raw segmentation
    filepath2 = "mia-result/2024-11-24-18-30-19/117122_SEG-PP.mha"  # Post-processed segmentation

    raw = sitk.ReadImage(filepath1)
    pp = sitk.ReadImage(filepath2)

    viewer = NiftiImageViewer(raw, pp, filepath1, filepath2)
    viewer.display()
