using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace SharpCV
{
    internal partial class cv2_native_api
    {
        /// <summary><c>cv::dnn::readNetFromDarknet(cfgFile, darknetModel)</c>.</summary>
        /// <param name="cfgFile">The .cfg network description.</param>
        /// <param name="darknetModel">The .weights file.</param>
        /// <param name="returnValue">Receives a new <c>cv::dnn::Net</c>; delete it with <see cref="dnn_Net_delete"/>.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus dnn_readNetFromDarknet(string cfgFile, string darknetModel, out IntPtr returnValue);

        /// <summary><c>new cv::dnn::Net()</c>: an empty network.</summary>
        /// <param name="net">Receives the network.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus dnn_Net_new(out IntPtr net);

        /// <summary><c>cv::dnn::readNetFromTensorflow(model, config)</c>.</summary>
        /// <param name="model">The frozen .pb graph.</param>
        /// <param name="config">The .pbtxt text graph, or null for none.</param>
        /// <param name="output">Receives the network.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus dnn_readNetFromTensorflow(string model, string config, out IntPtr output);

        /// <summary><c>cv::dnn::readNetFromTorch(model, isBinary)</c> (Torch7 .t7 files).</summary>
        /// <param name="model">The model file.</param>
        /// <param name="isBinary">True for the binary format; marshalled as the export's <c>int</c>.</param>
        /// <param name="output">Receives the network.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus dnn_readNetFromTorch(string model, [MarshalAs(UnmanagedType.Bool)] bool isBinary, out IntPtr output);

        /// <summary><c>cv::dnn::readNetFromONNX(onnxFile)</c>.</summary>
        /// <param name="model">The .onnx file.</param>
        /// <param name="output">Receives the network.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus dnn_readNetFromONNX(string model, out IntPtr output);

        /// <summary><c>cv::dnn::readTensorFromONNX(path)</c>: a TensorProto file as a Mat.</summary>
        /// <param name="path">The .pb tensor file.</param>
        /// <param name="output">Receives a new Mat.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus dnn_readTensorFromONNX(string path, out IntPtr output);

        /// <summary><c>delete net</c>.</summary>
        /// <param name="net">The network; never use it again.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus dnn_Net_delete(IntPtr net);

        /// <summary><c>cv::dnn::blobFromImage(*image, scalefactor, size, mean, swapRB, crop)</c>: an NCHW float32
        ///     blob.</summary>
        /// <param name="image">The image (a cv::Mat pointer, not an argument wrapper).</param>
        /// <param name="scalefactor">The multiplier applied after the mean subtraction.</param>
        /// <param name="size">The spatial output size; (0, 0) keeps the image's.</param>
        /// <param name="mean">The per-channel values subtracted first.</param>
        /// <param name="swapRB">True swaps the first and third channels; marshalled as the export's <c>int</c>.</param>
        /// <param name="crop">True crops after an aspect-preserving resize; marshalled as <c>int</c>.</param>
        /// <param name="output">Receives a new 4-D Mat.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus dnn_blobFromImage(IntPtr image, double scalefactor, Size size, Scalar mean,
            [MarshalAs(UnmanagedType.Bool)] bool swapRB, [MarshalAs(UnmanagedType.Bool)] bool crop, out IntPtr output);

        /// <summary><c>net-&gt;setInput(*blob, name)</c>. OpenCV 4.13 copies the blob into the input layer
        ///     (<c>blob_.copyTo(inputsData[...])</c> in net_impl.cpp), so the blob may be released right after the
        ///     call.</summary>
        /// <param name="net">The network.</param>
        /// <param name="input">The blob (a cv::Mat pointer).</param>
        /// <param name="name">The input layer's name; empty or null for the first input.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus dnn_Net_setInput(IntPtr net, IntPtr input, string name);

        /// <summary><c>net-&gt;forward(outputName)</c>. The returned header SHARES the network's output buffer, which the
        ///     next forward pass overwrites; <see cref="Net.forward"/> clones it, as opencv-python's conversion does.</summary>
        /// <param name="net">The network.</param>
        /// <param name="outputName">The output layer's name; empty or null for the last layer.</param>
        /// <param name="output">Receives a new header over the layer's output.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus dnn_Net_forward1(IntPtr net, string outputName, out IntPtr output);
    }
}
