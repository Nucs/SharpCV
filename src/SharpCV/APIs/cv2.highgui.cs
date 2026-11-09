using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace SharpCV
{
    public partial class cv_api
    {
        /// <summary><c>cv2.imshow(winname, mat)</c>: shows the image in a window (created on first use). Needs a desktop
        ///     session and a <see cref="waitKey"/> call to paint.</summary>
        /// <param name="windownName">The window title.</param>
        /// <param name="img">The image (a Mat or an NDArray; the window keeps its own copy of the pixels).</param>
        /// <exception cref="ArgumentNullException"><paramref name="img"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">OpenCV cannot show the image (no GUI backend, unsupported type), or cv2
        ///     refuses the NDArray.</exception>
        public void imshow(string windownName, InputArray img)
        {
            using var mat = img.Open("mat");
            // highgui_imshow takes the cv::Mat itself; Handle refuses a disposed Mat instead of passing OpenCV null.
            var handle = mat.Mat.Handle;
            NativeMethods.Call(() => cv2_native_api.highgui_imshow(windownName, handle));
        }

        /// <summary><c>cv2.namedWindow(winname, flags)</c>.</summary>
        /// <param name="name">The window title.</param>
        /// <param name="mode">The window flags.</param>
        /// <exception cref="OpenCVException">OpenCV has no GUI backend.</exception>
        public void namedWindow(string name, WindowFlags mode)
        {
            NativeMethods.Call(() => cv2_native_api.highgui_namedWindow(name, (int)mode));
        }

        /// <summary><c>cv2.resizeWindow(winname, width, height)</c>.</summary>
        /// <param name="name">The window title.</param>
        /// <param name="width">The new width.</param>
        /// <param name="height">The new height.</param>
        /// <exception cref="OpenCVException">OpenCV has no GUI backend.</exception>
        public void resizeWindow(string name, int width, int height)
        {
            NativeMethods.Call(() => cv2_native_api.highgui_resizeWindow(name, width, height));
        }

        /// <summary><c>cv2.waitKey(delay)</c>: processes window events for <paramref name="delay"/> ms.</summary>
        /// <param name="delay">Milliseconds to wait; 0 waits for a key.</param>
        /// <returns>The pressed key's code, or -1 when none was pressed (it returned nothing before SharpCV 0.15).</returns>
        /// <exception cref="OpenCVException">OpenCV has no GUI backend.</exception>
        public int waitKey(int delay = 0)
        {
            int output = -1;
            NativeMethods.Call(() => cv2_native_api.highgui_waitKey(delay, out output));
            return output;
        }

        /// <summary><c>cv2.destroyAllWindows()</c>.</summary>
        /// <exception cref="OpenCVException">OpenCV has no GUI backend.</exception>
        public void destroyAllWindows()
        {
            NativeMethods.Call(() => cv2_native_api.highgui_destroyAllWindows());
        }
    }
}
