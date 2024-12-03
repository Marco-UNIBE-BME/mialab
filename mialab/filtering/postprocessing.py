"""The post-processing module contains classes for image filtering mostly applied after a classification.

Image post-processing aims to alter images such that they depict a desired representation.
"""
import warnings

import numpy as np
import pymia.filtering
import pymia.filtering.filter as pymia_fltr
import SimpleITK as sitk
import math

# Our dictionary keys
BINARY_IMAGE_KEY:str = 'image'
CLOSING_KERNEL_SIZE:str = 'cks'
OPENING_KERNEL_SIZE:str = 'oks'
# MEDIAN_KERNEL_SIZE:str = 'mks'
PROCESSED_IM_KEY:str = 'final'
CONTOURS_KEY:str = 'contour'
#etc.

class ImagePostProcessing(pymia_fltr.Filter):
    """Represents a post-processing filter."""

    def __init__(self):
        """Initializes a new instance of the ImagePostProcessing class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: pymia_fltr.FilterParams = None) -> sitk.Image:
        """Registers an image.

        Args:
            image (sitk.Image): The image.
            params (FilterParams): The parameters.

        Returns:
            sitk.Image: The post-processed image.
        """

        # todo: replace this filter by a post-processing - or do we need post-processing at all?

        utils:PostProcessingUtils = PostProcessingUtils()

        """Step 1: Initialize a data strcuture and populate it with individual binary images for all labels."""
        data:dict[dict] = utils.init_data_structure(image) # NOTE: We omit label 0. The background is not being post-processed.
        orig_image = image
        #image = utils.upsample_volume(image, 2)
        utils.split_image(image, data)
        # utils.calculate_contours(data)

        """Step 2: Populate data structure with metadata for processing in Step 3."""
        #warnings.warn("Function calc_oc_kernels is being overriden manually for testing. ~marco")
        utils.calc_co_kernels(data)

        """Step 3: Application of our 'per-label' post-processing."""
        morph = MorphologicalOperations()

        # Erosion:
        # eroded_gray_matter = morph.erosion(image=data[2][BINARY_IMAGE_KEY])
        # data[2][BINARY_IMAGE_KEY] = eroded_gray_matter

        # Dilation:
        # dilated_gray_matter = morph.dilation(image=data[2][PROCESSED_IM_KEY])
        # data[2][PROCESSED_IM_KEY] = dilated_gray_matter

        # Closing:
        # for label in data.keys():
        #     data[label][PROCESSED_IM_KEY] = morph.closing(image=data[label][BINARY_IMAGE_KEY],
        #                                                   closing_radius=data[label][CLOSING_KERNEL_SIZE],
        #                                                   kernel_type=sitk.sitkBall)

        # Opening:
        # for label in data.keys():
        #     data[label][PROCESSED_IM_KEY] = morph.opening(image=data[label][BINARY_IMAGE_KEY],
        #                                                   opening_radius=data[label][OPENING_KERNEL_SIZE],
        #                                                   kernel_type=sitk.sitkBall)

        """Uncomment here to have the Closing and Opening."""
        # Closing and Opening:
        # for label in data.keys():
        #     processed_image:sitk.Image = morph.closing_opening(image=data[label][BINARY_IMAGE_KEY],
        #                                                closing_radius=data[label][CLOSING_KERNEL_SIZE],
        #                                                opening_radius=data[label][OPENING_KERNEL_SIZE],
        #                                                kernel_type=sitk.sitkBall)
        #     data[label][PROCESSED_IM_KEY] = processed_image

        # Black hat transform:
        # for label in data.keys():
        #     data[label][PROCESSED_IM_KEY] = data[label][BINARY_IMAGE_KEY] + morph.black_top_hat(data[label][BINARY_IMAGE_KEY])

        # Hole-Filling:
        # for label in data.keys():
        #     data[label][PROCESSED_IM_KEY] = morph.binary_fill_hole(data[label][BINARY_IMAGE_KEY])

        # Hole-Filling by Marco:
        # for label in data.keys():
        #     # Retrieve the binary mask for the label
        #     binary_image = data[label][BINARY_IMAGE_KEY]
        #
        #     # Apply hole-filling
        #     # hole_filling_filter = sitk.BinaryFillholeImageFilter()
        #     # hole_filled_image = hole_filling_filter.Execute(binary_image)
        #
        #     # Thresholded
        #     hole_filled_image = morph.fill_small_holes_2d(binary_image, 200)
        #
        #     # Update the processed image
        #     data[label][PROCESSED_IM_KEY] = hole_filled_image

        # Opening and Hole-Filling by Marco:
        for label in data.keys():
            processed_image = data[label][BINARY_IMAGE_KEY]
            processed_image = morph.opening(image=processed_image,
                                            opening_radius=data[label][OPENING_KERNEL_SIZE],
                                            kernel_type=sitk.sitkBall)
            # processed_image = morph.fill_small_holes_2d(processed_image, 50)
            data[label][PROCESSED_IM_KEY] = processed_image

        """Uncomment here to get Median filtering only."""
        # Median filtering:
        # medianf = sitk.MedianImageFilter()
        # medianf.SetRadius(3)
        # for label in data.keys():
        #     processed_image = medianf.Execute(data[label][BINARY_IMAGE_KEY])
        #     processed_image = utils.downsample_volume(processed_image, orig_image)
        #     data[label][PROCESSED_IM_KEY] = processed_image

        # output_image = medianf.Execute(image)
        # return utils.downsample_volume(output_image, orig_image)
        # return medianf.Execute(image)

        """Uncomment here to have Opening and Median filtering."""
        # Opening and Median filtering:
        # medianf = sitk.MedianImageFilter()
        # medianf.SetRadius(2)
        # for label in data.keys():
        #     processed_image = data[label][BINARY_IMAGE_KEY]
        #     processed_image = medianf.Execute(processed_image)
        #     processed_image = morph.opening(image=processed_image,
        #                                     opening_radius=data[label][OPENING_KERNEL_SIZE],
        #                                     kernel_type=sitk.sitkBall)
        #     data[label][PROCESSED_IM_KEY] = processed_image

        """Step 4: Stitching our processed per label images back into one image."""
        output_image = sitk.Image(orig_image.GetSize(), sitk.sitkUInt8)
        output_image.CopyInformation(orig_image)

        for label in data.keys():
            processed_image = data[label][PROCESSED_IM_KEY]
            #processed_image = utils.downsample_volume(processed_image, orig_image)
            label_mask = sitk.Mask(image = sitk.Cast(output_image == 0, sitk.sitkUInt8) * label,
                                     maskImage=processed_image,
                                     outsideValue=0)
            
            output_image += label_mask

        # for label in data.keys():
        #     contour_image = data[label][CONTOURS_KEY]
        #     labeled_contour = sitk.Mask(
        #         sitk.Cast(output_image == 0, sitk.sitkUInt8) * label, 
        #         maskImage=contour_image,
        #         outsideValue=0
        #     )

        #     output_image += labeled_contour

        """STep 5: Apply post-post-processing."""
        # Create a binary mask of the locations where the output_image is 0
        # zero_mask = sitk.Cast(output_image == 0, sitk.sitkUInt8)

        # # Combine original_image values with output_image values where zero_mask is True
        # masked_output_image = sitk.Mask(
        #     image=orig_image,
        #     maskImage=zero_mask,
        #     outsideValue=0
        # )

        # # Replace the values in output_image where zero_mask is 1
        # output_image = sitk.Cast(output_image, orig_image.GetPixelID())
        # output_image += masked_output_image
        # zero_mask = sitk.Cast(output_image == 0, sitk.sitkUInt8)
        #
        # # Combine original_image values with output_image values where zero_mask is True
        # masked_output_image = sitk.Mask(
        #     image=orig_image,
        #     maskImage=zero_mask,
        #     outsideValue=0
        # )
        #
        # # Replace the values in output_image where zero_mask is 1
        # output_image = sitk.Cast(output_image, orig_image.GetPixelID())
        # output_image += masked_output_image
        # mf = sitk.MedianImageFilter()
        # mf.SetRadius([1,1,1])
        # output_image =  mf.Execute(output_image)
        # return  mf.Execute(output_image)
        # output_image = utils.downsample_volume(output_image, orig_image)
        # warnings.warn('No post-processing implemented. Can you think about something?')

        return output_image

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'ImagePostProcessing:\n' \
            .format(self=self)

class PostProcessingUtils:
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

    def get_surface_to_volume(self, binary_image:sitk.Image) -> float:
        """This function claculates the surface to volume ration for a binarized labeled voxel volume (i.e. a label image)"""
        # Calculate surface-to-volume ratio and fragmentation
        binary_array = sitk.GetArrayFromImage(binary_image)
        contour_image = sitk.LabelContour(binary_image)
        contour_array = sitk.GetArrayFromImage(contour_image)
        surface_voxels = np.sum(contour_array)
        total_voxels = np.sum(binary_array)
        return surface_voxels / total_voxels if total_voxels > 0 else 0
    
    def get_num_fragments(self, binary_image) -> int:
        connected_components = sitk.ConnectedComponent(binary_image)
        return len(np.unique(sitk.GetArrayFromImage(connected_components))) - 1

    def upsample_volume(self, volume:sitk.Image, factor:float):
        """This function upsamples a volume by a given factor"""
        original_spacing = volume.GetSpacing()
        original_size = volume.GetSize()

        # Compute new spacing and size
        new_spacing = [s / factor for s in original_spacing]
        new_size = [int(sz * factor) for sz in original_size]

        # Resample
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputOrigin(volume.GetOrigin())
        resampler.SetOutputDirection(volume.GetDirection())
        resampler.SetInterpolator(sitk.sitkLinear)  # Linear interpolation for upsampling
        return resampler.Execute(volume)
    
    def downsample_volume(self, volume:sitk.Image, original_volume:sitk.Image):
        """THis function downsamples a volume to match the size deinition of an original image"""
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(original_volume.GetSpacing())
        resampler.SetSize(original_volume.GetSize())
        resampler.SetOutputOrigin(original_volume.GetOrigin())
        resampler.SetOutputDirection(original_volume.GetDirection())
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Nearest neighbor for binary data
        return resampler.Execute(volume)
    
    def calculate_contours(self, data: dict) -> None:
        """
        Calculates the contours for each label's binary image and stores them in the data dictionary.
        
        Args:
            data (dict): A dictionary where keys are labels and values are dicts containing binary images.
            contour_key (str): The key under which the contour images will be stored in the data dictionary.
        """
        for label, label_data in data.items():
            if label == 0:
                continue

            # Calculate the contour using SimpleITK's LabelContour function
            binary_image = label_data[BINARY_IMAGE_KEY]
            contour_image:sitk.Image = sitk.LabelContour(binary_image)

            # Store the contour in the data dictionary
            label_data[CONTOURS_KEY] = contour_image

            # Debugging: Print size and origin to verify consistency
            print(f"Label {label}: Contour calculated. Size = {contour_image.GetSize()}, Origin = {contour_image.GetOrigin()}, Contour Voxels = {np.sum(sitk.GetArrayFromImage(contour_image))}")

    def calc_co_kernels(self, data: dict) -> None:
        """
        Calculates the opening and closing kernel sizes for each label based on structural complexity and anatomical structure.

        Args:
            data (dict): A dictionary where keys are labels and values are dicts containing binary images
                         and other label-specific information.
        """
        for label in data.keys():
            if BINARY_IMAGE_KEY not in data[label]:
                continue

            # Access the binary mask for the label
            binary_image = data[label][BINARY_IMAGE_KEY]
            binary_array = sitk.GetArrayFromImage(binary_image)

            # Skip empty labels
            if np.sum(binary_array) == 0:
                data[label][OPENING_KERNEL_SIZE] = 1
                data[label][CLOSING_KERNEL_SIZE] = 1
                print(f"Label {label}: Empty label, skipping.")
                continue

            surface_to_volume_ratio = self.get_surface_to_volume(binary_image)
            num_fragments = self.get_num_fragments(binary_image)

            # Conditional approach:
            # Assign kernel sizes based on calculated features
            if surface_to_volume_ratio >= 0.30 and num_fragments > 750:  # With UpSamp: 8000 else 750
                # High complexity and fragmentation -> small kernel size
                closing_kernel_size = 0
                opening_kernel_size = 1  # With UpSamp: 1 else 1
            elif surface_to_volume_ratio < 0.30 and num_fragments < 500:  # With UpSamp: 1000 else 500
                # Low complexity and fragmentation -> big kernel size
                closing_kernel_size = 0
                opening_kernel_size = 3  # With UpSamp: 5 else 3
            else:
                # Intermediate complexity and fragmentation -> medium kernel size
                closing_kernel_size = 0
                opening_kernel_size = 2  # With UpSamp: 3 else 2

            # # Heuristic approach:
            # # Sigmoid steepness parameter
            # k = 10  # Controls the smoothness of the transitions
            #
            # # Safe exponential to avoid overflow
            # def safe_exp(x):
            #     return math.exp(max(-700, min(700, x)))
            #
            # # High complexity contribution
            # high_score = (
            #         1 / (1 + safe_exp(-k * (surface_to_volume_ratio - 0.30))) *
            #         1 / (1 + safe_exp(-k * (num_fragments - 750)))
            # )
            #
            # # Low complexity contribution
            # low_score = (
            #         (1 - 1 / (1 + safe_exp(-k * (surface_to_volume_ratio - 0.30)))) *
            #         (1 - 1 / (1 + safe_exp(-k * (num_fragments - 500))))
            # )
            #
            # # Combine scores into kernel size
            # opening_kernel_size = 1 + low_score - high_score
            #
            # # Round to nearest integer and clip to valid range
            # opening_kernel_size = max(1, min(3, round(opening_kernel_size)))
            # closing_kernel_size = 0

            # Debugging: Print metrics and kernel sizes
            print(f"Label {label}: Surface-to-Volume Ratio = {surface_to_volume_ratio:.2f}, "
                  f"Fragments = {num_fragments}, "
                  f"OPENING_KERNEL_SIZE = {opening_kernel_size}, CLOSING_KERNEL_SIZE = {closing_kernel_size}")

            # Assign kernel sizes to the data structure
            data[label][OPENING_KERNEL_SIZE] = opening_kernel_size
            data[label][CLOSING_KERNEL_SIZE] = closing_kernel_size

    def rank_labels_amount_voxels(self, prediction_image:sitk.Image, data:dict) -> None:
        from pymia.filtering.postprocessing import LargestNConnectedComponents
        pass
        # n = len(data)
        # lncc = LargestNConnectedComponents(number_of_components=n, consecutive_component_labels=True)
        # im:sitk.Image = lncc.execute(image=prediction_image)

class MorphologicalOperations():  # José: New morphological opening and closing post-processing
    """Represents a collection of morphological operation filters."""

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

    def erosion(self, image:sitk.Image, erosion_radius:int=1) -> sitk.Image:
        """Apply erosion operation to an image."""
        self._erosion_filter.SetKernelRadius(erosion_radius)
        print(self._erosion_filter.GetKernelType())
        return self._erosion_filter.Execute(image)

    def dilation(self, image:sitk.Image, dilation_radius:int=1) -> sitk.Image:
        """Apply dilation operation to an image."""
        self._dilation_filter.SetKernelRadius(dilation_radius)
        return self._dilation_filter.Execute(image)

    def closing(self, image: sitk.Image, closing_radius:int, kernel_type:int=sitk.sitkBall) -> sitk.Image:
        """Apply morphological closing to an image."""
        self._closing_filter.SetKernelRadius(closing_radius)
        self._closing_filter.SetKernelType(kernel_type)
        return self._closing_filter.Execute(image)

    def opening(self, image: sitk.Image, opening_radius:int, kernel_type:int=sitk.sitkBall) -> sitk.Image:
        """Apply morphological opening to an image."""
        self._opening_filter.SetKernelRadius(opening_radius)
        self._opening_filter.SetKernelType(kernel_type)
        return self._opening_filter.Execute(image)

    def closing_opening(self, image: sitk.Image, closing_radius:int, opening_radius:int, kernel_type:int=sitk.sitkBall) -> sitk.Image:
        """Execute the morphological opening and closing operations."""
        im = self.closing(image, closing_radius, kernel_type)
        im = self.opening(im, opening_radius, kernel_type)
        return im

    def binary_fill_hole(self, image:sitk.Image) -> sitk.Image:
        # self._binary_hole_filling_filter.SetFullyConnected(True)
        # print(self.binary_hole_filling_filter.GetFullyConnected())
        return self._binary_hole_filling_filter.Execute(image)
    
    def white_top_hat(self, image:sitk.Image) -> sitk.Image:
        return self._white_top_hat_filter.Execute(image)

    def black_top_hat(self, image:sitk.Image) -> sitk.Image:
        return self._black_top_hat_filter.Execute(image)


    def fill_small_holes_2d(self, binary_image, size_threshold):
        hole_filled_image = sitk.Image(binary_image.GetSize(), sitk.sitkUInt8)
        hole_filled_image.CopyInformation(binary_image)
        
        # Process each slice independently
        for z in range(binary_image.GetDepth()):
            slice_image = binary_image[:, :, z]
            inverted_slice = sitk.Not(slice_image)
            
            # Get connected components
            connected_components = sitk.ConnectedComponent(inverted_slice)
            
            # Relabel connected components
            relabel_filter = sitk.RelabelComponentImageFilter()
            relabeled_components = relabel_filter.Execute(connected_components)
            
            # Get the number of connected components
            num_labels = relabel_filter.GetNumberOfObjects()
            
            # Fill small holes in the slice
            small_holes_mask = sitk.Image(slice_image.GetSize(), sitk.sitkUInt8)
            small_holes_mask.CopyInformation(slice_image)
            for i in range(1, num_labels + 1):
                component_mask = sitk.Equal(relabeled_components, i)
                if sitk.GetArrayViewFromImage(component_mask).sum() <= size_threshold:
                    small_holes_mask = sitk.Or(small_holes_mask, component_mask)
            
            filled_slice = sitk.Or(slice_image, small_holes_mask)
            hole_filled_image = sitk.Paste(hole_filled_image, filled_slice, filled_slice.GetSize(), destinationIndex=[0, 0, z])
        
        return hole_filled_image

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