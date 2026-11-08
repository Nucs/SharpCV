using System;
using System.Collections.Generic;
using System.Text;

namespace SharpCV
{
    /// <summary>
    /// https://docs.opencv.org/master/d0/d3a/classcv_1_1DataType.html
    /// https://gist.github.com/pabloduque0/456cb62d02a027d3a458f31cdec6771d
    /// CV_&lt;bit-depth&gt;{U|S|F}C(&lt;number_of_channels&gt;)
    ///
    /// A Mapping of Type to Numbers in OpenCV
    ///         C1  C2  C3  C4
    /// CV_8U	0	8	16	24
    /// CV_8S	1	9	17	25
    /// CV_16U	2	10	18	26
    /// CV_16S	3	11	19	27
    /// CV_32S	4	12	20	28
    /// CV_32F	5	13	21	29
    /// CV_64F	6	14	22	30
    /// CV_16F	7	15	23	31
    /// </summary>
    /// <remarks>
    ///     A type packs the depth (<see cref="MatDepth"/>, the low 3 bits) and the channel count minus one (the bits
    ///     above), as OpenCV's <c>CV_MAKETYPE(depth, cn) = depth + ((cn - 1) &lt;&lt; 3)</c> does; any channel count up to
    ///     <c>CV_CN_MAX</c> (512) is a valid type even without a named member here. Read the parts with
    ///     <see cref="MatTypeExtensions.Depth"/> and <see cref="MatTypeExtensions.Channels"/>.
    /// </remarks>
    public enum MatType
    {
        /// <summary>No type; as a <c>ddepth</c> argument (-1) it means "the source's depth".</summary>
        Unknown = -1,
        /// <summary>8-bit unsigned, 1 channel (a grayscale image; NumSharp uint8).</summary>
        CV_8UC1 = 0,
        /// <summary>8-bit unsigned, 2 channels.</summary>
        CV_8UC2 = 8,
        /// <summary>8-bit unsigned, 3 channels (a BGR image).</summary>
        CV_8UC3 = 16,
        /// <summary>8-bit unsigned, 4 channels (BGRA).</summary>
        CV_8UC4 = 24,
        /// <summary>8-bit signed, 1 channel (NumSharp int8).</summary>
        CV_8SC1 = 1,
        /// <summary>8-bit signed, 2 channels.</summary>
        CV_8SC2 = 9,
        /// <summary>8-bit signed, 3 channels.</summary>
        CV_8SC3 = 17,
        /// <summary>8-bit signed, 4 channels.</summary>
        CV_8SC4 = 25,
        /// <summary>16-bit unsigned, 1 channel (NumSharp uint16).</summary>
        CV_16UC1 = 2,
        /// <summary>16-bit unsigned, 2 channels.</summary>
        CV_16UC2 = 10,
        /// <summary>16-bit unsigned, 3 channels.</summary>
        CV_16UC3 = 18,
        /// <summary>16-bit unsigned, 4 channels.</summary>
        CV_16UC4 = 26,
        /// <summary>16-bit signed, 1 channel (NumSharp int16).</summary>
        CV_16SC1 = 3,
        /// <summary>16-bit signed, 2 channels.</summary>
        CV_16SC2 = 11,
        /// <summary>16-bit signed, 3 channels.</summary>
        CV_16SC3 = 19,
        /// <summary>16-bit signed, 4 channels.</summary>
        CV_16SC4 = 27,
        /// <summary>32-bit signed, 1 channel (NumSharp int32).</summary>
        CV_32SC1 = 4,
        /// <summary>32-bit signed, 2 channels (a contour's points).</summary>
        CV_32SC2 = 12,
        /// <summary>32-bit signed, 3 channels.</summary>
        CV_32SC3 = 20,
        /// <summary>32-bit signed, 4 channels (a contour hierarchy).</summary>
        CV_32SC4 = 28,
        /// <summary>32-bit float, 1 channel (NumSharp float32; dnn blobs).</summary>
        CV_32FC1 = 5,
        /// <summary>32-bit float, 2 channels.</summary>
        CV_32FC2 = 13,
        /// <summary>32-bit float, 3 channels.</summary>
        CV_32FC3 = 21,
        /// <summary>32-bit float, 4 channels.</summary>
        CV_32FC4 = 29,
        /// <summary>64-bit float, 1 channel (NumSharp float64; affine matrices).</summary>
        CV_64FC1 = 6,
        /// <summary>64-bit float, 2 channels.</summary>
        CV_64FC2 = 14,
        /// <summary>64-bit float, 3 channels.</summary>
        CV_64FC3 = 22,
        /// <summary>64-bit float, 4 channels.</summary>
        CV_64FC4 = 30,
        /// <summary>16-bit float, 1 channel (NumSharp float16 / <see cref="Half"/>).</summary>
        CV_16FC1 = 7,
        /// <summary>16-bit float, 2 channels.</summary>
        CV_16FC2 = 15,
        /// <summary>16-bit float, 3 channels.</summary>
        CV_16FC3 = 23,
        /// <summary>16-bit float, 4 channels.</summary>
        CV_16FC4 = 31,
    }

    /// <summary>
    ///     OpenCV's element depths (<c>CV_8U</c> ... <c>CV_16F</c>), each the counterpart of exactly one NumSharp dtype:
    ///     uint8, int8, uint16, int16, int32, float32, float64, float16. These eight are the only dtypes a Mat can hold,
    ///     which is why an NDArray of any other dtype must be cast (int64 / uint64, as opencv-python does) or is refused.
    /// </summary>
    public enum MatDepth
    {
        /// <summary>8-bit unsigned (uint8).</summary>
        CV_8U = 0,
        /// <summary>8-bit signed (int8).</summary>
        CV_8S = 1,
        /// <summary>16-bit unsigned (uint16).</summary>
        CV_16U = 2,
        /// <summary>16-bit signed (int16).</summary>
        CV_16S = 3,
        /// <summary>32-bit signed (int32).</summary>
        CV_32S = 4,
        /// <summary>32-bit float (float32).</summary>
        CV_32F = 5,
        /// <summary>64-bit float (float64).</summary>
        CV_64F = 6,
        /// <summary>16-bit float (float16).</summary>
        CV_16F = 7,
    }

    /// <summary>
    ///     Splits and builds <see cref="MatType"/> values the way OpenCV's <c>CV_MAT_DEPTH</c>, <c>CV_MAT_CN</c> and
    ///     <c>CV_MAKETYPE</c> macros do.
    /// </summary>
    public static class MatTypeExtensions
    {
        /// <summary>The largest channel count a Mat type can encode (<c>CV_CN_MAX</c>).</summary>
        public const int CV_CN_MAX = 512;

        /// <summary><c>CV_MAT_DEPTH(type)</c>: the element depth of <paramref name="type"/>.</summary>
        /// <param name="type">The Mat type; <see cref="MatType.Unknown"/> is not a type.</param>
        /// <returns>Its depth.</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is negative.</exception>
        public static MatDepth Depth(this MatType type)
        {
            if (type < 0)
                throw new ArgumentOutOfRangeException(nameof(type), type, "MatType.Unknown has no depth.");
            return (MatDepth)((int)type & 7);
        }

        /// <summary><c>CV_MAT_CN(type)</c>: the channel count of <paramref name="type"/> (1 to <see cref="CV_CN_MAX"/>).</summary>
        /// <param name="type">The Mat type.</param>
        /// <returns>Its channel count.</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is negative.</exception>
        public static int Channels(this MatType type)
        {
            if (type < 0)
                throw new ArgumentOutOfRangeException(nameof(type), type, "MatType.Unknown has no channels.");
            return (((int)type >> 3) & (CV_CN_MAX - 1)) + 1;
        }

        /// <summary><c>CV_MAKETYPE(depth, cn)</c>: the type of <paramref name="channels"/> channels of
        ///     <paramref name="depth"/> (<c>CV_8U</c>, 3 is <see cref="MatType.CV_8UC3"/>).</summary>
        /// <param name="depth">The element depth.</param>
        /// <param name="channels">The channel count, 1 to <see cref="CV_CN_MAX"/>.</param>
        /// <returns>The type.</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="channels"/> is outside 1 to 512.</exception>
        public static MatType MakeType(this MatDepth depth, int channels)
        {
            if (channels < 1 || channels > CV_CN_MAX)
                throw new ArgumentOutOfRangeException(nameof(channels), channels, $"A Mat holds 1 to {CV_CN_MAX} channels.");
            return (MatType)((int)depth + ((channels - 1) << 3));
        }

        /// <summary><c>CV_ELEM_SIZE1</c>: the bytes of one channel value of <paramref name="depth"/>.</summary>
        /// <param name="depth">The depth.</param>
        /// <returns>1, 2, 4 or 8.</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="depth"/> is not one of the eight depths.</exception>
        public static int ElemSize1(this MatDepth depth) => depth switch
        {
            MatDepth.CV_8U or MatDepth.CV_8S => 1,
            MatDepth.CV_16U or MatDepth.CV_16S or MatDepth.CV_16F => 2,
            MatDepth.CV_32S or MatDepth.CV_32F => 4,
            MatDepth.CV_64F => 8,
            _ => throw new ArgumentOutOfRangeException(nameof(depth), depth, "Not an OpenCV depth."),
        };
    }
}
