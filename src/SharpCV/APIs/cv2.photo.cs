using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace SharpCV
{
    public partial class cv_api
    {
        /// <summary>
        /// Image Denoising
        /// Non-local Means Denoisin
        /// works with a single grayscale images
        /// https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_photo/py_non_local_means/py_non_local_means.html
        /// </summary>
        /// <param name="src">An 8-bit grayscale image (<c>cv2.fastNlMeansDenoising(src, None, h, templateWindowSize,
        ///     searchWindowSize)</c>).</param>
        /// <param name="h">The filter strength: higher removes more noise and more detail.</param>
        /// <param name="templateWindowSize">The patch size (odd).</param>
        /// <param name="searchWindowSize">The search window size (odd); time grows with its square.</param>
        /// <returns>A new image.</returns>
        /// <exception cref="OpenCVException">The image type is not supported.</exception>
        public Mat fastNlMeansDenoising(Mat src,
            float h = 3,
            int templateWindowSize = 7,
            int searchWindowSize = 21)
        {
            var dst = new Mat();
            NativeMethods.Call(() => cv2_native_api.photo_fastNlMeansDenoising(src.InputArray, dst.OutputArray, h, templateWindowSize, searchWindowSize));
            GC.KeepAlive(src);
            return dst;
        }

        /// <summary><c>cv2.fastNlMeansDenoisingColored(src, None, h, hColor, templateWindowSize, searchWindowSize)</c>:
        ///     denoises a colour image in the L*a*b* space.</summary>
        /// <param name="src">An 8-bit 3-channel image.</param>
        /// <param name="h">The luminance filter strength.</param>
        /// <param name="hColor">The colour filter strength.</param>
        /// <param name="templateWindowSize">The patch size (odd).</param>
        /// <param name="searchWindowSize">The search window size (odd).</param>
        /// <returns>A new image.</returns>
        /// <exception cref="OpenCVException">The image is not 8-bit 3-channel.</exception>
        public Mat fastNlMeansDenoisingColored(Mat src,
            float h = 3,
            float hColor = 3,
            int templateWindowSize = 7,
            int searchWindowSize = 21)
        {
            var dst = new Mat();
            NativeMethods.Call(() => cv2_native_api.photo_fastNlMeansDenoisingColored(src.InputArray, dst.OutputArray, h, hColor, templateWindowSize, searchWindowSize));
            GC.KeepAlive(src);
            return dst;
        }
    }
}
