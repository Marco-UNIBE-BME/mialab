"""The post-processing module contains classes for image filtering mostly applied after a classification.

Image post-processing aims to alter images such that they depict a desired representation.
"""
import warnings

import numpy as np
# import pydensecrf.densecrf as crf
# import pydensecrf.utils as crf_util
import pymia.filtering.filter as pymia_fltr
import SimpleITK as sitk

# Our dictionary keys
BINARY_IMAGE_KEY:str = 'image'
OPENING_KERNEL_SIZE:str = 'oks'
CLOSING_KERNEL_SIZE:str = 'cks'
OP_CL_IMAGE_KEY:str = 'ocik'
#etc.

class ImagePostProcessing(pymia_fltr.Filter):
    """Represents a post-processing filter."""

    def __init__(self):
        """Initializes a new instance of the ImagePostProcessing class."""
        super().__init__()
        # self.dense_crf:DenseCRF = DenseCRF()

    def execute(self, image: sitk.Image, params: pymia_fltr.FilterParams = None) -> sitk.Image:
        """Registers an image.

        Args:
            image (sitk.Image): The image.
            params (FilterParams): The parameters.

        Returns:
            sitk.Image: The post-processed image.
        """

        # todo: replace this filter by a post-processing - or do we need post-processing at all?

        prepost = PrePostProcessing()
        morph = MorphologicalOpeningClosing()

        # Step 1: Split image into labels
        data = prepost.init(image)
        prepost.get_label_binary_images(image, data)
        #prepost.define_kernel_sizes(data)

        if not data:
            raise ValueError("No labels found in the image for post-processing.")

        output_image = sitk.Image(image.GetSize(), sitk.sitkUInt8)
        output_image.CopyInformation(image)

        for label in data.keys():
            if label == 0:  # Skip the background label
                continue
        if BINARY_IMAGE_KEY not in data[label]:
            raise KeyError(f"Key '{BINARY_IMAGE_KEY}' missing for label {label}.")
        processed_image = morph.execute(data[label][BINARY_IMAGE_KEY])
        data[label][OP_CL_IMAGE_KEY] = processed_image
        output_image = sitk.Mask(
            image=output_image,
            maskImage=sitk.Cast(sitk.BinaryNot(processed_image), sitk.sitkUInt8),
            outsideValue=int(label)
        )

        #print(data[0][OPENING_KERNEL_SIZE])


        # Step 2 to N: Apply our post-processing on splited images

        # Step N+1: Merge them together

        # Step N+2: Post post processing


        # warnings.warn('No post-processing implemented. Can you think about something?')

        return image

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'ImagePostProcessing:\n' \
            .format(self=self)

class PrePostProcessing:
    def __init__(self) -> None:
        """Intiialize the pre-post-processor. Pass the image with predicted labels to intitialize the data structure for the pipeline."""
        pass

    def init(self, prediction_image:sitk.Image) -> dict:
        image_arr = sitk.GetArrayViewFromImage(prediction_image).flatten()
        labels = np.unique(image_arr)
        output_dict:dict = {}
        for label in labels:
            output_dict[label] = {}

        return output_dict

    def get_label_binary_images(self, image:sitk.Image, data:dict) -> None:
        """This function splits the predicted segmentation into individual binary images and stores the in a data structure."""
        for label in data.keys():
            if label == 0:
                continue
            binary_image = sitk.Equal(image, int(label))
            data[label][BINARY_IMAGE_KEY] = binary_image

    def define_kernel_sizes(self, data:dict) -> None:
        """This function defines the opening and closing kernel sizes per label image. (Data driven)"""
        pass

class MorphologicalOpeningClosing(pymia_fltr.Filter):  # José: New morphological opening and closing post-processing
    """Represents a morphological opening and closing filter."""

    def __init__(self, closing_radius: int = 1, opening_radius: int = 1):
        super().__init__()
        self.closing_radius = closing_radius
        self.opening_radius = opening_radius

        self.closing_filter = sitk.BinaryMorphologicalClosingImageFilter()
        self.closing_filter.SetKernelRadius(self.closing_radius)

        self.opening_filter = sitk.BinaryMorphologicalOpeningImageFilter()
        self.opening_filter.SetKernelRadius(self.opening_radius)

    def apply_closing(self, image: sitk.Image) -> sitk.Image:
        """Apply morphological closing to an image."""
        return self.closing_filter.Execute(image)

    def apply_opening(self, image: sitk.Image) -> sitk.Image:
        """Apply morphological opening to an image."""
        return self.opening_filter.Execute(image)

    def execute(self, image: sitk.Image, params: pymia_fltr.FilterParams = None) -> sitk.Image:
        """Execute the morphological opening and closing operations."""
        image = self.apply_closing(image)
        image = self.apply_opening(image)
        return image

    def execute_per_label(self, image: sitk.Image, params: pymia_fltr.FilterParams = None) -> sitk.Image:
        """Execute the morphological opening and closing operations."""
        # Making sure we have binary images before running binary closing and opening
        image_arr = sitk.GetArrayViewFromImage(image).flatten()
        labels = np.unique(image_arr)

        output_image = sitk.Image(image.GetSize(), sitk.sitkUInt8)
        output_image.CopyInformation(image)

        for label in labels:
            if label == 0: # Skip background
                continue
            print(label, type(label))
            binary_image = sitk.Equal(image, int(label))

            processed_image = self.apply_closing(binary_image)
            processed_image = self.apply_opening(processed_image)

            # TODO: Figure out priority. Via probabilities?
            output_image = sitk.Mask(image=output_image, maskImage=sitk.Cast(sitk.BinaryNot(processed_image), sitk.sitkUInt8), outsideValue=int(label))

        print(output_image.GetPixelIDTypeAsString())
        return output_image