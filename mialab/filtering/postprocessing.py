"""The post-processing module contains classes for image filtering mostly applied after a classification.

Image post-processing aims to alter images such that they depict a desired representation.
"""
import warnings

import numpy as np
import pymia.filtering
import pymia.filtering.filter as pymia_fltr
import SimpleITK as sitk

# Our dictionary keys
BINARY_IMAGE_KEY:str = 'image'
OPENING_KERNEL_SIZE:str = 'oks'
CLOSING_KERNEL_SIZE:str = 'cks'
PROCESSED_IM_KEY:str = 'final'
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
        # if params is None:
        #     raise ValueError('Parameters for DenseCRF are required for post-processing.')
        prepost:PrePostProcessing = PrePostProcessing()


        """Step 1: Initialize a data strcuture and populate it with individual binary images for all labels."""
        data:dict[dict] = prepost.init_data_structure(image) # NOTE: We omit label 0. The background is not being post-processed.
        prepost.split_image(image, data)

        """Step 2: Populate data structure with metadata for processing in Step 3."""
        prepost.calc_oc_kernels(data)

        """Step 3: Application of our 'per-label' post-processing."""
        morph = MorphologicalOpeningClosing()
        # return morph.binary_fill_hole(data[1][BINARY_IMAGE_KEY])

        for label in data.keys():
            data[label][PROCESSED_IM_KEY] = morph.apply_closing(data[label][BINARY_IMAGE_KEY], kernel_type=sitk.sitkCross)

        # Black hat transform
        # for label in data.keys():
        #     data[label][PROCESSED_IM_KEY] = data[label][BINARY_IMAGE_KEY] + morph.black_top_hat(data[label][BINARY_IMAGE_KEY])

        # Hole-Filling
        # for label in data.keys():
        #     data[label][PROCESSED_IM_KEY] = morph.binary_fill_hole(data[label][BINARY_IMAGE_KEY])

        # Erosion
        # eroded_gray_matter = morph.erosion(image=data[2][BINARY_IMAGE_KEY])
        # data[2][BINARY_IMAGE_KEY] = eroded_gray_matter

        # Opening and Closing
        # for label in data.keys():
        #     processed_image:sitk.Image = morph.closing_opening(image=data[label][BINARY_IMAGE_KEY],
        #                                                closing_radius=data[label][CLOSING_KERNEL_SIZE],
        #                                                opening_radius=data[label][OPENING_KERNEL_SIZE])
        #     data[label][PROCESSED_IM_KEY] = processed_image

        # Dilation
        # dilated_gray_matter = morph.dilation(image=data[2][PROCESSED_IM_KEY])
        # data[2][PROCESSED_IM_KEY] = dilated_gray_matter

        """Step 4: Stitching our processed per label images back into one image."""
        output_image = sitk.Image(image.GetSize(), sitk.sitkUInt8)
        output_image.CopyInformation(image)

        for label in data.keys():
            processed_image = data[label][PROCESSED_IM_KEY]
            output_image = sitk.Mask(image=output_image, maskImage=sitk.Cast(sitk.BinaryNot(processed_image), sitk.sitkUInt8), outsideValue=int(label))

        """STep 5: Apply post-post-processing."""
        # warnings.warn('No post-processing implemented. Can you think about something?')

        return output_image

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
    
    def init_data_structure(self, prediction_image:sitk.Image) -> dict:
        """This function intiializes the post processing data strcuture from a pipeline output image. i.e. the predicted labels."""
        image_arr = sitk.GetArrayViewFromImage(prediction_image).flatten()
        labels = np.unique(image_arr)
        output_dict:dict = {}
        for label in labels:
            if label == 0:
                continue
            output_dict[label] = {}

        return output_dict

    def split_image(self, image:sitk.Image, data:dict) -> None:
        """This function splits the predicted segmentation into individual binary images and stores the in a data structure."""
        for label in data.keys():
            binary_image = sitk.Equal(image, int(label))
            data[label][BINARY_IMAGE_KEY] = binary_image

    def calc_oc_kernels(self, data:dict) -> None:
        """This function defines the opening and closing kernel sizes per label image. (Data driven)"""
        for label in data.keys():
            data[label][OPENING_KERNEL_SIZE] = 1 # TODO: Make this data driven
            data[label][CLOSING_KERNEL_SIZE] = 1 # TODO: Make this data driven

    def rank_labels_amount_voxels(self, prediction_image:sitk.Image, data:dict) -> None:
        from pymia.filtering.postprocessing import LargestNConnectedComponents
        pass
        # n = len(data)
        # lncc = LargestNConnectedComponents(number_of_components=n, consecutive_component_labels=True)
        # im:sitk.Image = lncc.execute(image=prediction_image)

class MorphologicalOpeningClosing():  # José: New morphological opening and closing post-processing
    """Represents a morphological opening and closing filter."""

    def __init__(self):
        super().__init__()
        self._closing_filter = sitk.BinaryMorphologicalClosingImageFilter()
        self._closing_filter.SetKernelRadius(1)
        self._closing_filter.SafeBorderOn()

        self._opening_filter = sitk.BinaryMorphologicalOpeningImageFilter()
        self._opening_filter.SetKernelRadius(1)

        self._erosion_filter = sitk.BinaryErodeImageFilter()
        self._erosion_filter.SetKernelRadius(1)

        self._dilation_filter = sitk.BinaryDilateImageFilter()
        self._dilation_filter.SetKernelRadius(1)
        self._dilation_filter.SetKernelType(sitk.sitkCross)

        self._binary_hole_filling_filter = sitk.BinaryFillholeImageFilter()

        self._white_top_hat_filter = sitk.WhiteTopHatImageFilter()
        self._white_top_hat_filter.SetKernelRadius(1)

        self._black_top_hat_filter = sitk.BlackTopHatImageFilter()
        self._black_top_hat_filter.SetKernelRadius(2)
        self._black_top_hat_filter.SafeBorderOn()
        self._black_top_hat_filter.SetKernelType(sitk.sitkCross)

        # self.kernels = [sitk.sitkBall, sitk.sitkBox, sitk.sitkCross, sitk.sitkAnnulus]

    def apply_closing(self, image: sitk.Image, closing_radius:int=1, kernel_type:int=sitk.sitkBall) -> sitk.Image:
        """Apply morphological closing to an image."""
        self._closing_filter.SetKernelRadius(closing_radius)
        self._closing_filter.SetKernelType(kernel_type)
        return self._closing_filter.Execute(image)

    def apply_opening(self, image: sitk.Image, opening_radius:int=1, kernel_type:int=sitk.sitkBall) -> sitk.Image:
        """Apply morphological opening to an image."""
        self._opening_filter.SetKernelRadius(opening_radius)
        self._opening_filter.SetKernelType(kernel_type)
        return self._opening_filter.Execute(image)

    def erosion(self, image:sitk.Image, erosion_radius:int=1) -> sitk.Image:
        """Apply erosion operation to an image."""
        self._erosion_filter.SetKernelRadius(erosion_radius)
        print(self._erosion_filter.GetKernelType())
        return self._erosion_filter.Execute(image)

    def dilation(self, image:sitk.Image, dilation_radius:int=1) -> sitk.Image:
        """Apply dilation operation to an image."""
        self._dilation_filter.SetKernelRadius(dilation_radius)
        return self._dilation_filter.Execute(image)

    def closing_opening(self, image: sitk.Image, closing_radius:int, opening_radius:int, kernel_type:int=sitk.sitkBall) -> sitk.Image:
        """Execute the morphological opening and closing operations."""
        im = self.apply_closing(image, closing_radius, kernel_type)
        im = self.apply_opening(image, opening_radius, kernel_type)
        return im

    def binary_fill_hole(self, image:sitk.Image) -> sitk.Image:
        # self.binary_hole_filling_filter.SetFullyConnected(True)
        # print(self.binary_hole_filling_filter.GetFullyConnected())
        return self._binary_hole_filling_filter.Execute(im)

    def white_top_hat(self, image:sitk.Image) -> sitk.Image:
        return self._white_top_hat_filter.Execute(image)

    def black_top_hat(self, image:sitk.Image) -> sitk.Image:
        return self._black_top_hat_filter.Execute(image)

    # def execute_per_label(self, image: sitk.Image, params: pymia_fltr.FilterParams = None) -> sitk.Image:
    #     """Execute the morphological opening and closing operations."""
    #     # Making sure we have binary images before running binary closing and opening
    #     image_arr = sitk.GetArrayViewFromImage(image).flatten()
    #     labels = np.unique(image_arr)

    #     output_image = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    #     output_image.CopyInformation(image)

    #     for label in labels:
    #         if label == 0: # Skip background
    #             continue
    #         print(label, type(label))
    #         binary_image = sitk.Equal(image, int(label))

    #         processed_image = self.apply_closing(binary_image)
    #         processed_image = self.apply_opening(processed_image)

    #         # TODO: Figure out priority. Via probabilities?
    #         output_image = sitk.Mask(image=output_image, maskImage=sitk.Cast(sitk.BinaryNot(processed_image), sitk.sitkUInt8), outsideValue=int(label))

    #     print(output_image.GetPixelIDTypeAsString())
    #     return output_image