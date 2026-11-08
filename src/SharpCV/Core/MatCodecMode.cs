namespace SharpCV
{
    /// <summary>
    ///     How a crossing between <see cref="Mat"/> and NumSharp's <c>NDArray</c> may share memory: NumSharp's interop
    ///     modes, the same three members as the pythonnet codec's <c>NumpyCodecMode</c> and ONNX Runtime's
    ///     <c>OnnxCodecMode</c>.
    /// </summary>
    /// <remarks>
    ///     A crossing that no copy can repair (a dtype OpenCV has no depth for, such as bool or complex) fails in every
    ///     mode. A <see cref="Mat"/> can always be viewed (its byte steps become NumSharp strides), so the modes matter for
    ///     the NDArray -&gt; Mat direction, where opencv-python's rules decide: the last axis must be contiguous and the
    ///     strides must not grow towards the last axis.
    /// </remarks>
    public enum MatCodecMode
    {
        /// <summary>
        ///     The default: share the memory when the layout allows it, otherwise copy, and never fail because of a
        ///     layout. This is what opencv-python does with an input array (a transposed, reversed, column-strided or
        ///     broadcast array is copied, an int64 array is cast to int32).
        /// </summary>
        Auto = 0,

        /// <summary>
        ///     Always share; a crossing that would need a copy throws <see cref="OpenCVException"/>. Use it to prove that a
        ///     hot loop copies nothing, and for arrays OpenCV must write INTO (opencv-python refuses an output array whose
        ///     layout needs a copy: "Layout of the output array img is incompatible with cv::Mat").
        /// </summary>
        View = 1,

        /// <summary>
        ///     Always copy, so later writes on either side never reach the other (a snapshot).
        /// </summary>
        Copy = 2,
    }
}
