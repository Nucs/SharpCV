using System;
using System.Collections.Generic;
using System.Text;

namespace SharpCV
{
    /// <summary>The <c>cv2.dnn</c> module: loads networks and turns images into blobs.</summary>
    public class Dnn
    {
        /// <summary><c>cv2.dnn.readNetFromTensorflow(model, config)</c>.</summary>
        /// <param name="model">The frozen .pb graph.</param>
        /// <param name="config">The .pbtxt text graph, or null.</param>
        /// <returns>The network.</returns>
        /// <exception cref="OpenCVException">The files cannot be read or parsed.</exception>
        public Net readNetFromTensorflow(string model, string config = null)
        {
            IntPtr handle = IntPtr.Zero;
            NativeMethods.Call(() => cv2_native_api.dnn_readNetFromTensorflow(model, config, out handle));
            return new Net(handle);
        }

        /// <summary><c>cv2.dnn.readNetFromTorch(model, isBinary)</c> (Torch7 .t7 files).</summary>
        /// <param name="model">The model file.</param>
        /// <param name="isBinary">True for the binary format.</param>
        /// <returns>The network.</returns>
        /// <exception cref="OpenCVException">The file cannot be read or parsed.</exception>
        public Net ReadNetFromTorch(string model, bool isBinary = true)
        {
            IntPtr handle = IntPtr.Zero;
            NativeMethods.Call(() => cv2_native_api.dnn_readNetFromTorch(model, isBinary, out handle));
            return new Net(handle);
        }

        /// <summary><c>cv2.dnn.readNetFromONNX(onnxFile)</c>.</summary>
        /// <param name="model">The .onnx file.</param>
        /// <returns>The network.</returns>
        /// <exception cref="OpenCVException">The file cannot be read, or uses operators OpenCV's importer lacks.</exception>
        public Net ReadNetFromONNX(string model)
        {
            IntPtr handle = IntPtr.Zero;
            NativeMethods.Call(() => cv2_native_api.dnn_readNetFromONNX(model, out handle));
            return new Net(handle);
        }

        /// <summary>
        /// Creates 4-dimensional blob from image.
        /// </summary>
        /// <param name="image">input image (with 1-, 3- or 4-channels).</param>
        /// <param name="scaleFactor">The multiplier applied after the mean subtraction; 1.0 as in cv2.</param>
        /// <param name="size">spatial size for output image; default (0, 0) keeps the image's</param>
        /// <param name="mean">scalar with mean values which are subtracted from channels.</param>
        /// <param name="swapRB">flag which indicates that swap first and last channels in 3-channel image is necessary;
        ///     false as in cv2 (it defaulted to true until SharpCV 0.20).</param>
        /// <param name="crop">flag which indicates whether image will be cropped after resize or not</param>
        /// <returns>A new NCHW float32 Mat; its <see cref="Mat.data"/> is a <c>(1, C, H, W)</c> NDArray view.</returns>
        /// <exception cref="OpenCVException">OpenCV rejects the image.</exception>
        public Mat blobFromImage(Mat image, double scaleFactor = 1.0, (int, int) size = default, Scalar? mean = null, bool swapRB = false, bool crop = false)
        {
            IntPtr handle = IntPtr.Zero;
            NativeMethods.Call(() => cv2_native_api.dnn_blobFromImage(image,
                scaleFactor,
                new Size(size.Item1, size.Item2),
                mean ?? new Scalar(),
                swapRB,
                crop,
                out handle));
            GC.KeepAlive(image);
            return new Mat(handle);
        }
    }
}
