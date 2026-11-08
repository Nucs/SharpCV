using System;
using System.IO;
using NumSharp;
using SharpCV;
using static SharpCV.Binding;

namespace AppConsole
{
    /// <summary>
    ///     SharpCV with NumSharp the way opencv-python is used with NumPy: images are NDArrays that share OpenCV's memory,
    ///     NumSharp slicing and arithmetic work on them directly, and cv2 functions take them back without copies.
    /// </summary>
    /// <remarks>
    ///     Each example opens an <see cref="NDScope"/>: every NDArray the method creates (the temporaries of NumSharp
    ///     expressions included) is disposed when the scope closes, NumSharp's own idiom for transient arrays. The video
    ///     loop disposes per frame instead, because a method-wide scope would keep every frame's temporaries until the
    ///     end.
    /// </remarks>
    internal static class NumSharpExamples
    {
        /// <summary>Runs every example against the repository's data folder and prints what each one shows.</summary>
        /// <param name="dataDir">The data folder.</param>
        public static void Run(string dataDir)
        {
            ZeroCopyViews(Path.Combine(dataDir, "solar.jpg"));
            MaskToContours(Path.Combine(dataDir, "invoice.jpg"));
            VideoIntoOneBuffer(Path.Combine(dataDir, "road.mp4"));
            NormalizeForDnn(Path.Combine(dataDir, "solar.jpg"));
        }

        /// <summary>The image is an NDArray view; slicing gives views; drawing on a slice draws on the image.</summary>
        /// <param name="file">An image.</param>
        private static void ZeroCopyViews(string file)
        {
            using var scope = NDScope.Open();
            NDArray img = cv2.imread(file);                         // (410, 730, 3) uint8, OpenCV's own pixels
            NDArray roi = img["50:150, 100:300"];                    // a NumSharp view: no copy
            cv2.rectangle(roi, (10, 10), (90, 60), (0, 0, 255), -1); // draws INTO img, as in cv2

            NDArray rgb = img["..., ::-1"];                          // channel-reversed view; cv2 copies it on input
            var gray = cv2.cvtColor(rgb, ColorConversionCodes.COLOR_RGB2GRAY);
            var mean = np.mean(img.reshape(-1, 3).astype(np.float64), axis: 0);
            Console.WriteLine($"[views] shape {img.Shape}, red block at img[60, 110] = {img[60, 110]}, BGR mean {mean}, gray {gray.shape}");
        }

        /// <summary>A NumSharp comparison builds the mask OpenCV traces.</summary>
        /// <param name="file">A document scan.</param>
        private static void MaskToContours(string file)
        {
            using var scope = NDScope.Open();
            NDArray gray = cv2.imread(file, IMREAD_COLOR.IMREAD_GRAYSCALE);
            var mask = (gray < 128).astype(np.uint8) * 255;          // NumSharp threshold: dark ink -> 255
            var (contours, hierarchy) = cv2.findContours(mask, RetrievalModes.RETR_EXTERNAL, ContourApproximationModes.CHAIN_APPROX_SIMPLE);
            long points = 0;
            foreach (var c in contours)
                points += c.data.shape[0];                           // each contour is an (N, 1, 2) int32 view
            Console.WriteLine($"[mask] {contours.Length} contours, {points} points, hierarchy {hierarchy.shape}");
        }

        /// <summary>Every frame is decoded into the same NDArray; NumSharp measures the motion between frames.</summary>
        /// <param name="file">A video.</param>
        private static void VideoIntoOneBuffer(string file)
        {
            using var video = cv2.VideoCapture(file);
            using var frame = np.zeros(new Shape(720, 1280, 3), np.uint8);
            NDArray previous = null;
            int count = 0;
            double motion = 0;
            while (video.read(frame).Item1)
            {
                var current = frame.astype(np.int16);                // widen before subtracting
                if (previous is not null)
                {
                    using var difference = current - previous;
                    using var magnitude = np.abs(difference);
                    using var average = np.mean(magnitude);
                    motion += (double)average;
                    previous.Dispose();
                }

                previous = current;
                count++;
            }

            previous?.Dispose();
            Console.WriteLine($"[video] {count} frames decoded into one buffer, mean motion {motion / Math.Max(1, count - 1):F2}");
        }

        /// <summary>NumSharp broadcasting does blobFromImage's work: scale, subtract the mean, HWC -&gt; NCHW.</summary>
        /// <param name="file">An image.</param>
        private static void NormalizeForDnn(string file)
        {
            using var scope = NDScope.Open();
            NDArray img = cv2.resize(cv2.imread(file), (224, 224));
            var mean = np.array(new float[] { 0.406f, 0.456f, 0.485f });             // BGR order
            var x = (img.astype(np.float32) / 255f - mean);                          // (224, 224, 3), broadcast
            var nchw = np.expand_dims(np.transpose(x, new[] { 2, 0, 1 }), 0);        // (1, 3, 224, 224)

            var blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (224, 224), (0.406 * 255, 0.456 * 255, 0.485 * 255));
            var diff = (double)np.max(np.abs(nchw - blob.data));
            Console.WriteLine($"[dnn] NumSharp blob {nchw.Shape} vs blobFromImage {blob.shape}: max difference {diff:E2}");
        }
    }
}
