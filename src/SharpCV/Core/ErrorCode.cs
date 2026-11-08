namespace SharpCV
{
    /// <summary>
    ///     OpenCV's error codes (<c>cv::Error::Code</c> in <c>opencv2/core/base.hpp</c>), the number opencv-python prints
    ///     in parentheses of a <c>cv2.error</c> message (<c>(-215:Assertion failed)</c>). Read it from
    ///     <see cref="OpenCVException.Status"/> to branch on the kind of failure instead of parsing the message text.
    /// </summary>
    public enum ErrorCode
    {
        /// <summary>Everything is OK (never raised).</summary>
        StsOk = 0,
        /// <summary>Pseudo error for back trace.</summary>
        StsBackTrace = -1,
        /// <summary>Unknown or unspecified error.</summary>
        StsError = -2,
        /// <summary>Internal error (bad state).</summary>
        StsInternal = -3,
        /// <summary>Insufficient memory.</summary>
        StsNoMem = -4,
        /// <summary>A function argument or parameter is bad; SharpCV's NDArray conversion refusals use it too, as
        ///     opencv-python's argument parser does.</summary>
        StsBadArg = -5,
        /// <summary>Unsupported function.</summary>
        StsBadFunc = -6,
        /// <summary>An iteration did not converge.</summary>
        StsNoConv = -7,
        /// <summary>Tracing.</summary>
        StsAutoTrace = -8,
        /// <summary>The image header is NULL.</summary>
        HeaderIsNull = -9,
        /// <summary>The image size is invalid.</summary>
        BadImageSize = -10,
        /// <summary>The offset is invalid.</summary>
        BadOffset = -11,
        /// <summary>Bad data pointer.</summary>
        BadDataPtr = -12,
        /// <summary>The image step is wrong; this may happen for a non-continuous matrix.</summary>
        BadStep = -13,
        /// <summary>Bad model or channel sequence.</summary>
        BadModelOrChSeq = -14,
        /// <summary>Bad number of channels, for example a function that accepts only single-channel matrices.</summary>
        BadNumChannels = -15,
        /// <summary>Bad number of channels (1U).</summary>
        BadNumChannel1U = -16,
        /// <summary>The input image depth is not supported by the function.</summary>
        BadDepth = -17,
        /// <summary>Bad alpha channel.</summary>
        BadAlphaChannel = -18,
        /// <summary>The number of dimensions is out of range.</summary>
        BadOrder = -19,
        /// <summary>Incorrect input origin.</summary>
        BadOrigin = -20,
        /// <summary>Incorrect input alignment.</summary>
        BadAlign = -21,
        /// <summary>Bad callback.</summary>
        BadCallBack = -22,
        /// <summary>Bad tile size.</summary>
        BadTileSize = -23,
        /// <summary>The input COI is not supported.</summary>
        BadCOI = -24,
        /// <summary>Incorrect input ROI.</summary>
        BadROISize = -25,
        /// <summary>The mask is tiled.</summary>
        MaskIsTiled = -26,
        /// <summary>Null pointer.</summary>
        StsNullPtr = -27,
        /// <summary>Incorrect vector length.</summary>
        StsVecLengthErr = -28,
        /// <summary>Incorrect filter structure content.</summary>
        StsFilterStructContentErr = -29,
        /// <summary>Incorrect transform kernel content.</summary>
        StsKernelStructContentErr = -30,
        /// <summary>Incorrect filter offset value.</summary>
        StsFilterOffsetErr = -31,
        /// <summary>The input/output structure size is incorrect.</summary>
        StsBadSize = -201,
        /// <summary>Division by zero.</summary>
        StsDivByZero = -202,
        /// <summary>In-place operation is not supported.</summary>
        StsInplaceNotSupported = -203,
        /// <summary>The request cannot be completed.</summary>
        StsObjectNotFound = -204,
        /// <summary>The formats of the input/output arrays differ.</summary>
        StsUnmatchedFormats = -205,
        /// <summary>A flag is wrong or not supported.</summary>
        StsBadFlag = -206,
        /// <summary>Bad CvPoint.</summary>
        StsBadPoint = -207,
        /// <summary>Bad mask format (neither 8uC1 nor 8sC1).</summary>
        StsBadMask = -208,
        /// <summary>The sizes of the input/output structures do not match.</summary>
        StsUnmatchedSizes = -209,
        /// <summary>The data format or type is not supported by the function.</summary>
        StsUnsupportedFormat = -210,
        /// <summary>A parameter is out of range.</summary>
        StsOutOfRange = -211,
        /// <summary>Invalid syntax or structure of a parsed file.</summary>
        StsParseError = -212,
        /// <summary>The requested function or feature is not implemented.</summary>
        StsNotImplemented = -213,
        /// <summary>An allocated block has been corrupted.</summary>
        StsBadMemBlock = -214,
        /// <summary>An assertion failed (<c>CV_Assert</c>); the most common code.</summary>
        StsAssert = -215,
        /// <summary>No CUDA support.</summary>
        GpuNotSupported = -216,
        /// <summary>GPU API call error.</summary>
        GpuApiCallError = -217,
        /// <summary>No OpenGL support.</summary>
        OpenGlNotSupported = -218,
        /// <summary>OpenGL API call error.</summary>
        OpenGlApiCallError = -219,
        /// <summary>OpenCL API call error.</summary>
        OpenCLApiCallError = -220,
        /// <summary>OpenCL double precision is not supported.</summary>
        OpenCLDoubleNotSupported = -221,
        /// <summary>OpenCL initialization error.</summary>
        OpenCLInitError = -222,
        /// <summary>No AMD BLAS / FFT for OpenCL.</summary>
        OpenCLNoAMDBlasFft = -223,
    }
}
