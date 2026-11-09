using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace SharpCV
{
    /// <summary>
    ///     The <c>flipCode</c> of <c>cv2.flip</c>: 1 mirrors left-right (around the y axis), 0 top-bottom (around the x
    ///     axis), -1 both.
    /// </summary>
    /// <remarks>
    ///     Until SharpCV 0.15 these were carotene's internal flags (3rdparty\carotene\include\carotene\types.hpp:
    ///     horizontal 1, vertical 2, both 3), but SharpCV passes the value straight to <c>cv::flip</c> as its flipCode, where
    ///     every positive value means horizontal: the vertical and both modes flipped horizontally.
    /// </remarks>
    public enum FLIP_MODE
    {
        /// <summary>Mirror left-right (cv2 flipCode 1).</summary>
        FLIP_HORIZONTAL_MODE = 1,
        /// <summary>Mirror top-bottom (cv2 flipCode 0).</summary>
        FLIP_VERTICAL_MODE = 0,
        /// <summary>Mirror both ways, a 180 degree rotation (cv2 flipCode -1).</summary>
        FLIP_BOTH_MODE = -1
    };
}
