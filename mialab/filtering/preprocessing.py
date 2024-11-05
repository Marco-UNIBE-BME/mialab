"""The pre-processing module contains classes for image pre-processing.

Image pre-processing aims to improve the image quality (image intensities) for subsequent pipeline steps.
"""
import warnings

import numpy as np
import pymia.filtering.filter as pymia_fltr
import SimpleITK as sitk


class ImageNormalization(pymia_fltr.Filter):
    """Represents a normalization filter."""

    def __init__(self):
        """Initializes a new instance of the ImageNormalization class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: pymia_fltr.FilterParams = None) -> sitk.Image:
        """Executes a normalization on an image.

        Args:
            image (sitk.Image): The image.
            params (FilterParams): The parameters (unused).

        Returns:
            sitk.Image: The normalized image.
        """

        img_arr = sitk.GetArrayFromImage(image)

        # todo: normalize the image using numpy
        # warnings.warn('No normalization implemented. Returning unprocessed image.')

        # José: Z-score normalization
        img_mean = np.mean(img_arr)
        img_std = np.std(img_arr)
        if img_std != 0:
            img_arr_normalized = (img_arr - img_mean) / img_std
        else:
            img_arr_normalized = img_arr  # José: Avoid division by zero if the image has no variance

        # José: Min-max normalization (alternative)
        # img_min = np.min(img_arr)
        # img_max = np.max(img_arr)
        # if img_max != img_min:
        #     img_arr_normalized = (img_arr - img_min) / (img_max - img_min)
        # else:
        #     img_arr_normalized = img_arr  # José: Avoid division by zero if the image is constant

        img_out = sitk.GetImageFromArray(img_arr_normalized)
        img_out.CopyInformation(image)

        return img_out

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'ImageNormalization:\n' \
            .format(self=self)


class SkullStrippingParameters(pymia_fltr.FilterParams):
    """Skull-stripping parameters."""

    def __init__(self, img_mask: sitk.Image):
        """Initializes a new instance of the SkullStrippingParameters

        Args:
            img_mask (sitk.Image): The brain mask image.
        """
        self.img_mask = img_mask


class SkullStripping(pymia_fltr.Filter):
    """Represents a skull-stripping filter."""

    def __init__(self):
        """Initializes a new instance of the SkullStripping class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: SkullStrippingParameters = None) -> sitk.Image:
        """Executes a skull stripping on an image.

        Args:
            image (sitk.Image): The image.
            params (SkullStrippingParameters): The parameters with the brain mask.

        Returns:
            sitk.Image: The normalized image. #NOTE: Was like this, probably copy paste error; Left it unchanged ~marco
        """
        mask = params.img_mask  # the brain mask

        # todo: remove the skull from the image by using the brain mask
        # warnings.warn('No skull-stripping implemented. Returning unprocessed image.')
        image = sitk.Mask(image, mask)

        return image

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'SkullStripping:\n' \
            .format(self=self)


class ImageRegistrationParameters(pymia_fltr.FilterParams):
    """Image registration parameters."""

    def __init__(self, atlas: sitk.Image, transformation: sitk.Transform, is_ground_truth: bool = False):
        """Initializes a new instance of the ImageRegistrationParameters

        Args:
            atlas (sitk.Image): The atlas image.
            transformation (sitk.Transform): The transformation for registration.
            is_ground_truth (bool): Indicates weather the registration is performed on the ground truth or not.
        """
        self.atlas = atlas
        self.transformation = transformation
        self.is_ground_truth = is_ground_truth


class ImageRegistration(pymia_fltr.Filter):
    """Represents a registration filter."""

    def __init__(self):
        """Initializes a new instance of the ImageRegistration class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: ImageRegistrationParameters = None) -> sitk.Image:
        """Registers an image.

        Args:
            image (sitk.Image): The image.
            params (ImageRegistrationParameters): The registration parameters.

        Returns:
            sitk.Image: The registered image.
        """

        # todo: replace this filter by a registration. Registration can be costly, therefore, we provide you the
        # transformation, which you only need to apply to the image!
        warnings.warn('No registration implemented. Returning unregistered image')

        atlas = params.atlas
        transform = params.transformation
        is_ground_truth = params.is_ground_truth  # the ground truth will be handled slightly different

        # note: if you are interested in registration, and want to test it, have a look at
        # pymia.filtering.registration.MultiModalRegistration. Think about the type of registration, i.e.
        # do you want to register to an atlas or inter-subject? Or just ask us, we can guide you ;-)

        return image

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'ImageRegistration:\n' \
            .format(self=self)

#### UNIT TEST
if __name__ == '__main__':
    print(4*"+", "Executing Unit test of preprocessing",4*"+")
    import matplotlib.pyplot as plt

    skull_strip = SkullStripping() # Skull stripper object

    # LOAD SOME MASK
    mask_path = "dataset/train/101107/Brainmasknative.nii.gz" # Each image has its mask already
    mask = sitk.ReadImage(mask_path)
    params = SkullStrippingParameters(mask)# Object that loads the mask for stripping

    # LOAD THE CORRESPONDING IMAGE
    path:str = "dataset/train/101107/T2native.nii.gz"
    image:sitk.Image = sitk.ReadImage(path)

    # APPLY THE SKULLSTRIPPPING
    masked_image = skull_strip.execute(image, params)

    # DISPLAY THE RESULT ON ONE SLICE
    slice_index = image.GetSize()[2] // 2

    slice_image = image[:,:,slice_index]
    slice_masked = masked_image[:,:,slice_index]

    slice_array = sitk.GetArrayViewFromImage(slice_image)
    slice_masked_array = sitk.GetArrayFromImage(slice_masked)

    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(slice_array, cmap='gray')
    ax2.imshow(slice_masked_array, cmap='gray')
    plt.show()