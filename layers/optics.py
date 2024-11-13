import abc

import tensorflow as tf
import numpy as np

from numpy.fft import ifftshift
import fractions
import poppy

##############################
# Helper functions
##############################

def transp_fft2d(a_tensor, dtype=tf.complex64):
    """Takes images of shape [batch_size, x, y, channels] and transposes them
    correctly for tensorflows fft2d to work.
    """
    # Tensorflow's fft only supports complex64 dtype
    a_tensor = tf.cast(a_tensor, tf.complex64)
    # Tensorflow's FFT operates on the two innermost (last two!) dimensions
    a_tensor_transp = tf.transpose(a_tensor, [0, 3, 1, 2])
    a_fft2d = tf.fft2d(a_tensor_transp)
    a_fft2d = tf.cast(a_fft2d, dtype)
    a_fft2d = tf.transpose(a_fft2d, [0, 2, 3, 1])
    return a_fft2d


def transp_ifft2d(a_tensor, dtype=tf.complex64):
    a_tensor = tf.transpose(a_tensor, [0, 3, 1, 2])
    a_tensor = tf.cast(a_tensor, tf.complex64)
    a_ifft2d_transp = tf.ifft2d(a_tensor)
    # Transpose back to [batch_size, x, y, channels]
    a_ifft2d = tf.transpose(a_ifft2d_transp, [0, 2, 3, 1])
    a_ifft2d = tf.cast(a_ifft2d, dtype)
    return a_ifft2d


def compl_exp_tf(phase, dtype=tf.complex64, name='complex_exp'):
    """Complex exponent via euler's formula, since Cuda doesn't have a GPU kernel for that.
    Casts to *dtype*.
    """
    phase = tf.cast(phase, tf.float64)
    return tf.add(tf.cast(tf.cos(phase), dtype=dtype),
                  1.j * tf.cast(tf.sin(phase), dtype=dtype),
                  name=name)


def attach_summaries(name, var, image=False, log_image=False):
    if image:
        tf.summary.image(name, var, max_outputs=3)
    if log_image and image:
        tf.summary.image(name + '_log', tf.log(var + 1e-12), max_outputs=3)
    tf.summary.scalar(name + '_mean', tf.reduce_mean(var))
    tf.summary.scalar(name + '_max', tf.reduce_max(var))
    tf.summary.scalar(name + '_min', tf.reduce_min(var))
    tf.summary.histogram(name + '_histogram', var)


def fftshift2d_tf(a_tensor):
    input_shape = a_tensor.shape.as_list()

    new_tensor = a_tensor
    for axis in range(1, 3):
        split = (input_shape[axis] + 1) // 2
        mylist = np.concatenate((np.arange(split, input_shape[axis]), np.arange(split)))
        new_tensor = tf.gather(new_tensor, mylist, axis=axis)
    return new_tensor


def ifftshift2d_tf(a_tensor):
    input_shape = a_tensor.shape.as_list()

    new_tensor = a_tensor
    for axis in range(1, 3):
        n = input_shape[axis]
        split = n - (n + 1) // 2
        mylist = np.concatenate((np.arange(split, n), np.arange(split)))
        new_tensor = tf.gather(new_tensor, mylist, axis=axis)
    return new_tensor


def psf2otf(input_filter, output_size):
    '''Convert 4D tensorflow filter into its FFT.

    :param input_filter: PSF. Shape (height, width, num_color_channels, num_color_channels)
    :param output_size: Size of the output OTF.
    :return: The otf.
    '''
    # pad out to output_size with zeros
    # circularly shift so center pixel is at 0,0
    print(input_filter.shape.as_list())
    fh, fw, _, _= input_filter.shape.as_list()
    

    if output_size[0] != fh:
        pad = (output_size[0] - fh) / 2

        if (output_size[0] - fh) % 2 != 0:
            pad_top = pad_left = int(np.ceil(pad))
            pad_bottom = pad_right = int(np.floor(pad))
        else:
            pad_top = pad_left = int(pad) + 1
            pad_bottom = pad_right = int(pad) - 1

        padded = tf.pad(input_filter, [[pad_top, pad_bottom],
                                       [pad_left, pad_right], [0, 0], [0, 0]], "CONSTANT")
    else:
        padded = input_filter

    padded = tf.transpose(padded, [2, 0, 1, 3])
    padded = ifftshift2d_tf(padded)
    padded = tf.transpose(padded, [1, 2, 0, 3])

    ## Take FFT
    tmp = tf.transpose(padded, [2, 3, 0, 1])
    if tmp.dtype == tf.complex64:
        tmp = tf.fft2d(tmp)
    else:
        tmp = tf.fft2d(tf.complex(tmp, 0.))
    return tf.transpose(tmp, [2, 3, 0, 1])


def img_psf_conv(img, psf, otf=None, adjoint=False, circular=False):
    '''Performs a convolution of an image and a psf in frequency space.

    :param img: Image tensor.
    :param psf: PSF tensor.
    :param otf: If OTF is already computed, the otf.
    :param adjoint: Whether to perform an adjoint convolution or not.
    :param circular: Whether to perform a circular convolution or not.
    :return: Image convolved with PSF.
    '''
    img = tf.convert_to_tensor(img)
    psf = tf.convert_to_tensor(psf)

    img_shape = img.shape.as_list()
    print('the shape of img is', img_shape)

    if not circular:
        #target_side_length = 2 * img_shape[1] #2res
        target_side_length = 768

        height_pad = (target_side_length - img_shape[1]) / 2 # 1/2 res
        width_pad = (target_side_length - img_shape[1]) / 2
        
        pad_top, pad_bottom = int(np.ceil(height_pad)), int(np.floor(height_pad))
        pad_left, pad_right = int(np.ceil(width_pad)), int(np.floor(width_pad))

        img = tf.pad(img, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], "SYMMETRIC")
        img_shape = img.shape.as_list() # 2res
        print('the shape of padded_img is', img_shape)
    img_fft = transp_fft2d(img)

    if otf is None:
        otf = psf2otf(psf, output_size=img_shape[1:3])
        otf = tf.transpose(otf, [2, 0, 1, 3])

    otf = tf.cast(otf, tf.complex64)
    img_fft = tf.cast(img_fft, tf.complex64)

    if adjoint:
        result = transp_ifft2d(img_fft * tf.conj(otf))  # tf.conj 计算共e复数
    else:
        result = transp_ifft2d(img_fft * otf)

    #result = tf.cast(tf.real(result), tf.float32)

    if not circular:
        result = result[:, pad_top:-pad_bottom, pad_left:-pad_right, :]

    return result

def area_downsampling_tf(input_image, target_side_length):
    input_shape = input_image.shape.as_list()
    input_image = tf.cast(input_image, tf.float32)

    if not input_shape[1] % target_side_length:
        factor = int(input_shape[1] / target_side_length)
        output_img = tf.nn.avg_pool(input_image,
                                    [1, factor, factor, 1],
                                    strides=[1, factor, factor, 1],
                                    padding="VALID")
    else:
        # We upsample the image and then average pool
        lcm_factor = least_common_multiple(target_side_length, input_shape[1]) / target_side_length

        if lcm_factor > 10:
            print(
                "Warning: area downsampling is very expensive and not precise if source and target wave length have a large least common multiple")
            upsample_factor = 10
        else:
            upsample_factor = int(lcm_factor)

        img_upsampled = tf.image.resize_nearest_neighbor(input_image,
                                                         size=2 * [upsample_factor * target_side_length])
        output_img = tf.nn.avg_pool(img_upsampled,
                                    [1, upsample_factor, upsample_factor, 1],
                                    strides=[1, upsample_factor, upsample_factor, 1],
                                    padding="VALID")

    return output_img


def get_intensities(input_field):
    return tf.square(tf.abs(input_field), name='intensities')


##################################
# Optical elements & Propagation
##################################

class Propagation(abc.ABC):
    def __init__(self,
                 input_shape,
                 distance,
                 discretization_size,
                 wave_lengths):
        self.input_shape = input_shape
        self.distance = distance
        self.wave_lengths = wave_lengths
        self.wave_nos = 2. * np.pi / wave_lengths
        self.discretization_size = discretization_size

    @abc.abstractmethod
    def _propagate(self, input_field):
        """Propagate an input field through the medium
        """

    def __call__(self, input_field):
        return self._propagate(input_field)


class FresnelPropagation(Propagation):
    def _propagate(self, input_field):
        _, M_orig, N_orig, _ = self.input_shape
        print("padded_input_shape: shape", self.input_shape)#padded_input_shape: shape [None, 716, 716, 3]
  
        Mpad = (M_orig // 2)
        Npad = (N_orig // 2)

        M = M_orig + 2 * Mpad
        N = N_orig + 2 * Npad
        padded_input_field = tf.pad(input_field,
                                    [[0, 0], [Mpad, Mpad], [Npad, Npad], [0, 0]], 'SYMMETRIC')
        print("padded_input_field: shape", padded_input_field.shape)#padded_input_field: shape (?, 1024, 1024, 3)
        [x, y] = np.mgrid[-N // 2:N // 2,
                 -M // 2:M // 2]
        [x, y] = np.mgrid[-N // 2:N // 2,
                 -M // 2:M // 2]
        # Spatial frequency
        fx = x / (self.discretization_size * N)  # max frequency = 1/(2*pixel_size)
        fy = y / (self.discretization_size * M)

        # We need to ifftshift fx and fy here, because ifftshift doesn't exist in TF.
        fx = ifftshift(fx)
        fy = ifftshift(fy)

        fx = fx[None, :, :, None]
        fy = fy[None, :, :, None]

        squared_sum = np.square(fx) + np.square(fy)
        '''
        Mpad = 2 * M_orig
        Npad = 2 * N_orig
        M = M_orig + 2 * Mpad
        N = N_orig + 2 * Npad
        padded_input_field = tf.pad(input_field,
                                    [[0, 0], [Mpad, Mpad], [Npad, Npad], [0, 0]])

        [x, y] = np.mgrid[-N // 2:N // 2,
                 -M // 2:M // 2]
        # Spatial frequency
        fx = x / (self.discretization_size * N)  # max frequency = 1/(2*pixel_size)
        fy = y / (self.discretization_size * M)

        # We need to ifftshift fx and fy here, because ifftshift doesn't exist in TF.
        fx = ifftshift(fx)
        fy = ifftshift(fy)

        fx = fx[None, :, :, None]
        fy = fy[None, :, :, None]

        squared_sum = np.square(fx) + np.square(fy)
        '''
       
        batch=1
        _, height, width, channels = padded_input_field.get_shape()
        if tf.contrib.framework.is_tensor(self.distance):
            tmp = np.float64(self.wave_lengths * np.pi * -1. * squared_sum)
            constant_exp_part_init = tf.constant_initializer(tmp)
            constant_exponent_part = tf.get_variable("Fresnel_kernel_constant_exponent_part",
                                                     initializer=constant_exp_part_init,
                                                     shape=padded_input_field.shape,
                                                     #shape=[None, width, height, channels],
                                                     dtype=tf.float64,
                                                     trainable=False)

            H = compl_exp_tf(self.distance * constant_exponent_part, dtype=tf.complex64,
                             name='fresnel_kernel')
        else:  # Save some memory
            tmp = np.float64(self.wave_lengths * np.pi * -1. * squared_sum * self.distance)
            constant_exp_part_init = tf.constant_initializer(tmp)
            constant_exponent_part = tf.get_variable("Fresnel_kernel_constant_exponent_part",
                                                     initializer=constant_exp_part_init,
                                                     #shape=padded_input_field.shape,
                                                      shape=[batch, width, height, channels],
                                                     dtype=tf.float64,
                                                     trainable=False)

            H = compl_exp_tf(constant_exponent_part, dtype=tf.complex64,
                             name='fresnel_kernel')

        objFT = transp_fft2d(padded_input_field)
        out_field = transp_ifft2d(tf.math.divide(objFT , H ))

        #return out_field, input_field, H, objFT
        return out_field[:, Mpad:-Mpad, Npad:-Npad, :], input_field, H, objFT
        print("shape of out_filed:", out_field.shape)
        print("shope of input_field:", input_field.shape)




def propagate_exact(input_field,
                    distance,
                    input_sample_interval,
                    wave_lengths):
    _, M_orig, N_orig, _ = input_field.shape.as_list()
    print("the shape of input_field is:", input_field.shape.as_list())
    # zero padding.
    Mpad = M_orig // 4
    Npad = N_orig // 4
    M = M_orig + 2 * Mpad
    N = N_orig + 2 * Npad
    padded_input_field = tf.pad(input_field,
                                [[0, 0], [Mpad, Mpad], [Npad, Npad], [0, 0]])

    [x, y] = np.mgrid[-N // 2:N // 2,
             -M // 2:M // 2]

    # Spatial frequency
    fx = x / (input_sample_interval * N)  # max frequency = 1/(2*pixel_size)
    fy = y / (input_sample_interval * M)

    # We need to ifftshift fx and fy here, because ifftshift doesn't exist in TF.
    fx = ifftshift(fx)
    fy = ifftshift(fy)

    fx = fx[None, :, :, None]
    fy = fy[None, :, :, None]

    # We create a non-trainable variable so that this computation can be reused
    # from call to call.
    if tf.contrib.framework.is_tensor(distance):
        tmp = np.float64(
            2 * np.pi * (1 / wave_lengths) * np.sqrt(1. - (wave_lengths * fx) ** 2 - (wave_lengths * fy) ** 2))
        constant_exp_part_init = tf.constant_initializer(tmp)
        constant_exponent_part = tf.get_variable("Fresnel_kernel_constant_exponent_part",
                                                 initializer=constant_exp_part_init,
                                                 shape=padded_input_field.shape,
                                                 dtype=tf.float64,
                                                 trainable=False)

        H = compl_exp_tf(distance * constant_exponent_part, dtype=tf.complex64,
                         name='fresnel_kernel')
    else:  # Save some memory
        tmp = np.float64(
            2 * np.pi * (distance / wave_lengths) * np.sqrt(1. - (wave_lengths * fx) ** 2 - (wave_lengths * fy) ** 2))
        constant_exp_part_init = tf.constant_initializer(tmp)
        constant_exponent_part = tf.get_variable("Fresnel_kernel_constant_exponent_part",
                                                 initializer=constant_exp_part_init,
                                                 shape=padded_input_field.shape,
                                                 dtype=tf.float64,
                                                 trainable=False)

        H = compl_exp_tf(constant_exponent_part, dtype=tf.complex64,
                         name='fresnel_kernel')

    objFT = transp_fft2d(padded_input_field)
    out_field = transp_ifft2d(objFT / H)
    print("the shape of out_field is:", out_field.shape.as_list())
    return out_field[:, Mpad:-Mpad, Npad:-Npad, :]


def propagate_fresnel(input_field,
                      distance,
                      sampling_interval,
                      wave_lengths):
    input_shape = input_field.shape.as_list()
    propagation = FresnelPropagation(input_shape,
                                     distance=distance,
                                     discretization_size=sampling_interval,
                                     wave_lengths=wave_lengths)
    return propagation(input_field)

class ZernikeSystem():
    def __init__(self,
                 back_distance,
                 zernike_volume,
                 wave_resolution,
                 wave_lengths,
                 sensor_distance,
                 sensor_resolution,
                 input_sample_interval,
                 refractive_idcs,
                 height_tolerance,
                 focal_length,
                 target_distance=None,
                 upsample=True,
                 depth_bins=None):
        '''Simulates a one-lens system with a zernike-parameterized lens.

        :param zernike_volume: Zernike basis functions.
                               Tensor of shape (num_basis_functions, wave_resolution[0], wave_resolution[1]).
        :param wave_resolution: Resolution of the simulated wavefront. Shape wave_resolution.
        :param wave_lengths: Wavelengths to be simulated. Shape (num_wavelengths).
        :param sensor_distance: Distance of sensor to optical element.
        :param sensor_resolution: Resolution of simulated sensor.
        :param input_sample_interval: Sampling interval of aperture. Scalar.
        :param refractive_idcs: Refractive idcs of simulated material at wave_lengths.
        :param height_tolerance: Manufacturing tolerance of element. Adds the respective level of noise to be robust to
                                 manufacturing imperfections.
        :param target_distance: Allows to define the depth of a PSF that will *always* be evaluated. That can then be
                                used for instance for EDOF deconvolution.
        :param upsample: Whether the image should be upsampled to the PSF resolution or the PSF should be downsampled
                         to the sensor resolution.
        :param depth_bins: Depths at which PSFs should be simulated.
        '''

        self.back_distance = back_distance
        self.sensor_distance = sensor_distance
        self.zernike_volume = zernike_volume
        self.wave_resolution = wave_resolution
        self.wave_lengths = wave_lengths
        self.depth_bins = depth_bins
        self.sensor_resolution = sensor_resolution
        self.upsample = upsample
        self.target_distance = target_distance
        self.zernike_volume = zernike_volume
        self.height_tolerance = height_tolerance
        self.input_sample_interval = input_sample_interval
        self.refractive_idcs = refractive_idcs
        self.focal_length = focal_length
        self.psf_resolution = self.sensor_resolution

    def get_zemax_img(self, input_img, name='zemax_psf_map'):
        field = input_img
        sensor_incident_field, field, H, objFT = propagate_fresnel(field, distance=self.back_distance, sampling_interval=self.input_sample_interval, wave_lengths=self.wave_lengths)

        liner_phase = np.load('1029_512_512_5.86_ap.npy', allow_pickle=True)        
        _, height, width, _ = input_img.shape.as_list()
        height_map_shape = [1, height, width, 1]
        VL_1_liner = liner_phase[0]
        wave1_liner_initializer = tf.constant_initializer(VL_1_liner)
        VL_1_phase = liner_phase[1]
        wave1_phase_initializer = tf.constant_initializer(VL_1_phase)
        VL_2_liner = liner_phase[2]
        wave2_liner_initializer = tf.constant_initializer(VL_2_liner)
        VL_2_phase = liner_phase[3]
        wave2_phase_initializer = tf.constant_initializer(VL_2_phase)
        VL_3_liner = liner_phase[4]
        wave3_liner_initializer = tf.constant_initializer(VL_3_liner)
        VL_3_phase = liner_phase[5]
        wave3_pahse_initializer = tf.constant_initializer(VL_3_phase)
        with tf.variable_scope(name, reuse=False):
            self.zemax_wave1_liner = tf.get_variable('zemax_wave1_liner',
                                                shape=[1,height, width,1],
                                                dtype=tf.float64,
                                                trainable=True,
                                                initializer=wave1_liner_initializer)
            tf.add_to_collection('Optical_Variables', self.zemax_wave1_liner)
            
            self.zemax_wave1_phase = tf.get_variable('zemax_wave1_phase',
                                                shape=[1,height, width,1],
                                                dtype=tf.float64,
                                                trainable=True,
                                                initializer=wave1_phase_initializer)
            tf.add_to_collection('Optical_Variables', self.zemax_wave1_phase)
            
            self.zemax_wave2_liner = tf.get_variable('zemax_wave2_liner',
                                                shape=[1,height, width,1],
                                                dtype=tf.float64,
                                                trainable=True,
                                                initializer=wave2_liner_initializer)
            tf.add_to_collection('Optical_Variables', self.zemax_wave2_liner)
             
            self.zemax_wave2_phase = tf.get_variable('zemax_wave2_phase',
                                                shape=[1,height, width,1],
                                                dtype=tf.float64,
                                                trainable=True,
                                                initializer=wave2_phase_initializer)
            tf.add_to_collection('Optical_Variables', self.zemax_wave2_phase)
            
            self.zemax_wave3_liner = tf.get_variable('zemax_wave3_liner',
                                                shape=[1,height, width,1],
                                                dtype=tf.float64,
                                                trainable=True,
                                                initializer=wave3_liner_initializer)
            tf.add_to_collection('Optical_Variables', self.zemax_wave3_liner)
            
            self.zemax_wave3_phase = tf.get_variable('zemax_wave3_phase',
                                                shape=[1,height, width,1],
                                                dtype=tf.float64,
                                                trainable=True,
                                                initializer=wave3_pahse_initializer)
            tf.add_to_collection('Optical_Variables', self.zemax_wave3_phase)
   
              
        phase = []
        amp = []
        amp =[self.zemax_wave1_liner, self.zemax_wave2_liner, self.zemax_wave3_liner]
        amp = tf.cast(tf.abs(amp), tf.complex64)
        phase = [self.zemax_wave1_phase, self.zemax_wave2_phase, self.zemax_wave3_phase]
        phase = compl_exp_tf(phase, dtype=tf.complex64)
        psf = tf.multiply(amp, phase)
        psf = tf.transpose(psf, [4,2,3,1,0])
        psf = tf.squeeze(psf, axis=0)
        #psf = self.element(psf)
        intensity_psf = get_intensities(psf)
        self.intensity_psf , self.psf, self.sensor_incident_field =  intensity_psf, psf, sensor_incident_field
        print("the shape of psf is:", psf.shape.as_list())
        
        
        wave_img = img_psf_conv(sensor_incident_field, psf)
        sensor_img = tf.abs(wave_img)  # [None,2048,2048,3]
        input_energy = tf.reduce_sum(tf.abs(input_img), axis=[1, 2], keep_dims=True)
        output_energy = tf.reduce_sum(sensor_img, axis=[1, 2], keep_dims=True)
        normalize_weight = input_energy / output_energy
        sensor_img = sensor_img * normalize_weight
        sensor_img = get_intensities(sensor_img)
        print("the shape of sensor_img is:", sensor_img.shape.as_list()) # now is 716
        if self.upsample:
            print("Images are upsampled to wave resolution")
            sensor_img = tf.image.resize_images(sensor_img, self.wave_resolution,
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return sensor_img


