"""The post-processing module contains classes for image filtering mostly applied after a classification.

Image post-processing aims to alter images such that they depict a desired representation.
"""
import warnings

import numpy as np
# import pydensecrf.densecrf as crf
# import pydensecrf.utils as crf_util
import pymia.filtering.filter as pymia_fltr
import SimpleITK as sitk


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
        # if params is None:
        #     raise ValueError('Parameters for DenseCRF are required for post-processing.')
        
        # post_processed = self.dense_crf(image, params)
        warnings.warn('No post-processing implemented. Can you think about something?')

        return image

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'ImagePostProcessing:\n' \
            .format(self=self)


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


            output_image = sitk.Mask(image=output_image, maskImage=sitk.Cast(sitk.BinaryNot(processed_image), sitk.sitkUInt8), outsideValue=int(label))

        print(output_image.GetPixelIDTypeAsString())
        return output_image