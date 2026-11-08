using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace SharpCV
{
    public partial class cv_api
    {
        /// <summary><c>cv2.VideoCapture(index, apiPreference)</c>: opens a camera.</summary>
        /// <param name="device">The camera index.</param>
        /// <param name="api">The backend; CAP_ANY lets OpenCV choose.</param>
        /// <returns>The capture; check <see cref="SharpCV.VideoCapture.isOpened"/>.</returns>
        /// <exception cref="OpenCVException">OpenCV failed to create the capture object.</exception>
        public VideoCapture VideoCapture(int device, VideoCaptureAPIs api = VideoCaptureAPIs.CAP_ANY)
        {
            IntPtr handle = IntPtr.Zero;
            NativeMethods.Call(() => cv2_native_api.videoio_VideoCapture_new3(device, (int)api, out handle));
            return new VideoCapture(handle);
        }

        /// <summary><c>cv2.VideoCapture(filename, apiPreference)</c>: opens a video file, image sequence or stream.</summary>
        /// <param name="fileName">The source.</param>
        /// <param name="api">The backend; CAP_ANY lets OpenCV choose (the FFmpeg plugin ships with the runtime
        ///     package).</param>
        /// <returns>The capture; check <see cref="SharpCV.VideoCapture.isOpened"/>.</returns>
        /// <exception cref="OpenCVException">OpenCV failed to create the capture object.</exception>
        public VideoCapture VideoCapture(string fileName, VideoCaptureAPIs api = VideoCaptureAPIs.CAP_ANY)
        {
            IntPtr handle = IntPtr.Zero;
            NativeMethods.Call(() => cv2_native_api.videoio_VideoCapture_new2(fileName, (int)api, out handle));
            return new VideoCapture(handle);
        }
    }
}
