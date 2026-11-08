using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using SharpCV;
using static SharpCV.Binding;

namespace UnitTest
{
    /// <summary>
    ///     The NDArray &lt;-&gt; Mat bridge against opencv-python's own conversion (<c>pyopencv_to</c> /
    ///     <c>pyopencv_from</c>): every dtype, every layout, the copy decision, the output-argument rules and the three
    ///     <see cref="MatCodecMode"/>s. Each case hands cv2 the very same array (a zero-copy numpy view with the same
    ///     strides), so both sides convert the identical layout.
    /// </summary>
    [TestClass]
    public class BridgeTests : Test
    {
        /// <summary>The dtypes NumPy and NumSharp share; cv2 accepts eight, casts two, refuses the rest.</summary>
        private static readonly string[] SharedDtypes =
            { "bool", "uint8", "int8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32", "float64", "complex128" };

        /// <summary>
        ///     Every shared dtype through <c>cv2.flip(x, 0)</c>: the same success (int64 / uint64 come back int32, as cv2
        ///     casts them) or the same refusal (bool, uint32, complex128).
        /// </summary>
        [TestMethod]
        public void dtype_matrix_matches_cv2()
        {
            foreach (var name in SharedDtypes)
            {
                var x = np.arange(12).reshape(3, 4).astype(np.dtype(name));
                CompareFlip(x, $"dtype {name}");
            }
        }

        /// <summary>NumSharp's own dtypes without a NumPy twin (char, decimal) are refused like cv2 refuses an unknown
        ///     dtype.</summary>
        [TestMethod]
        public void numsharp_only_dtypes_are_refused()
        {
            foreach (var dtype in new[] { np.dtype(typeof(char)), np.dtype(typeof(decimal)) })
            {
                var x = np.zeros(new Shape(3, 4), dtype);
                var e = Assert.ThrowsException<OpenCVException>(() => new Mat(x));
                Assert.AreEqual(ErrorCode.StsBadArg, e.Status);
                StringAssert.Contains(e.Message, "is not supported");
            }
        }

        /// <summary>The layouts cv2 meets in practice (and the edge ones), uint8 and float32: the same flip result, or
        ///     the same refusal.</summary>
        [TestMethod]
        public void layout_matrix_matches_cv2()
        {
            foreach (var dtype in new[] { np.uint8, np.float32 })
                foreach (var (what, x) in Layouts(dtype))
                    CompareFlip(x, $"{dtype.name} {what}");
        }

        /// <summary>
        ///     cv2 refuses an OUTPUT array whose layout would need a copy ("Layout of the output array img is incompatible
        ///     with cv::Mat") or that is read-only, and draws into every other one; drawing a rectangle on each layout
        ///     therefore exposes cv2's copy decision, and SharpCV's NDArray overload must decide the same and draw the
        ///     same pixels into the caller's memory.
        /// </summary>
        [TestMethod]
        public void output_layout_rules_match_cv2()
        {
            foreach (var dtype in new[] { np.uint8, np.float32 })
            {
                foreach (var (what, make) in OutputLayouts(dtype))
                {
                    // Each side draws on its own fresh array built the same way.
                    var (mineBase, mine) = make();
                    var (theirsBase, theirs) = make();
                    Exception sharpError = null;
                    try
                    {
                        cv2.rectangle(mine, (0, 0), (1, 1), (7, 7, 7), -1);
                    }
                    catch (OpenCVException e)
                    {
                        sharpError = e;
                    }

                    Oracle.Run(() =>
                    {
                        Exception pythonError = null;
                        try
                        {
                            using var _ = (Python.Runtime.PyObject)Oracle.cv2.rectangle(theirs, (0, 0), (1, 1), (7, 7, 7), -1);
                        }
                        catch (Python.Runtime.PythonException e)
                        {
                            pythonError = e;
                        }

                        Assert.AreEqual(pythonError is null, sharpError is null,
                            $"{dtype.name} {what}: cv2 {(pythonError is null ? "drew" : "refused: " + pythonError.Message)}, SharpCV {(sharpError is null ? "drew" : "refused: " + sharpError.Message)}");
                        if (pythonError is null)
                            Oracle.AssertSame(theirsBase, mineBase, $"{dtype.name} {what}: pixels drawn into the caller's array");
                    });
                }
            }
        }

        /// <summary>View shares the array's memory (OpenCV's writes reach it), Copy never does, and Auto shares when the
        ///     layout allows it.</summary>
        [TestMethod]
        public void modes_share_or_copy()
        {
            var nd = np.zeros(new Shape(4, 6), np.uint8);
            using (var view = new Mat(nd, MatCodecMode.View))
                cv2.rectangle(view, (0, 0), (2, 2), 9, -1);
            Assert.AreEqual(9, (byte)nd[1, 1], "View: OpenCV wrote into the NDArray");

            var nd2 = np.zeros(new Shape(4, 6), np.uint8);
            using (var copy = new Mat(nd2, MatCodecMode.Copy))
                cv2.rectangle(copy, (0, 0), (2, 2), 9, -1);
            Assert.AreEqual(0, (byte)nd2[1, 1], "Copy: the NDArray is untouched");

            var nd3 = np.zeros(new Shape(4, 6), np.uint8);
            using (var auto = new Mat(nd3))
                cv2.rectangle(auto, (0, 0), (2, 2), 9, -1);
            Assert.AreEqual(9, (byte)nd3[1, 1], "Auto on a viewable layout shares");

            // A layout that needs a copy: Auto copies silently (cv2's input behaviour), View refuses.
            var transposed = np.zeros(new Shape(6, 4), np.uint8).T;
            using (var auto = new Mat(transposed))
                Assert.AreEqual(MatType.CV_8UC1, auto.MatType);
            var e = Assert.ThrowsException<OpenCVException>(() => new Mat(transposed, MatCodecMode.View));
            Assert.AreEqual(ErrorCode.StsBadArg, e.Status);
        }

        /// <summary>
        ///     Mat -&gt; NDArray: every depth with 1 to 4 channels views as cv2 shapes its arrays (dims, then channels),
        ///     with the dtype of the depth, and an ROI's view keeps the parent's row stride like a numpy slice.
        /// </summary>
        [TestMethod]
        public void mat_views_have_cv2_shapes_dtypes_and_strides()
        {
            foreach (var name in new[] { "uint8", "int8", "uint16", "int16", "int32", "float32", "float64", "float16" })
            {
                for (int channels = 1; channels <= 4; channels++)
                {
                    var shape = channels == 1 ? new Shape(5, 7) : new Shape(5, 7, channels);
                    var nd = np.arange(shape.size).reshape(shape).astype(np.dtype(name));
                    using var mat = new Mat(nd, MatCodecMode.Copy);
                    Assert.AreEqual(channels, mat.Channels, $"{name} x{channels}: channels");
                    var view = mat.data;
                    Assert.AreEqual(name, view.dtype.name, $"{name} x{channels}: dtype");
                    CollectionAssert.AreEqual(nd.shape, view.shape, $"{name} x{channels}: shape");
                    Assert.IsTrue(np.array_equal(nd, view), $"{name} x{channels}: values");
                }
            }

            var img = cv2.imread(imgSolar);
            var roi = img[(10, 20), (5, 15)];
            var roiView = roi.data;
            CollectionAssert.AreEqual(new long[] { 730 * 3, 3, 1 }, roiView.strides, "ROI strides are the parent's");
            Oracle.Run(() =>
            {
                using var sliced = Oracle.at(Oracle.cv2.imread(imgSolar), Oracle.builtins.slice(10, 20), Oracle.builtins.slice(5, 15));
                Oracle.AssertSame(sliced, roiView, "ROI view");
                // numpy's own strides of the slice (a tuple of ints) against the view's.
                using var pyStrides = sliced.GetAttr("strides");
                var strides = new long[pyStrides.Length()];
                for (int i = 0; i < strides.Length; i++)
                {
                    using var item = pyStrides[i];
                    strides[i] = item.As<long>();
                }

                CollectionAssert.AreEqual(strides, roiView.strides, "numpy slice strides");
            });
        }

        /// <summary>An n-D blob (dnn input) keeps its dimensions: a 3-D array is not folded into channels by
        ///     Net.setInput, while the generic conversion folds it (cv2's CV_ND flag).</summary>
        [TestMethod]
        public void dnn_blob_is_n_dimensional()
        {
            var blob = cv2.dnn.blobFromImage(cv2.imread(imgSolar), 1 / 255.0, (64, 48), swapRB: true);
            var view = blob.data;
            CollectionAssert.AreEqual(new long[] { 1, 3, 48, 64 }, view.shape);
            Assert.AreEqual("float32", view.dtype.name);

            NDArray img = cv2.imread(imgSolar);
            Oracle.Run(() => Oracle.AssertSame(Oracle.cv2.dnn.blobFromImage(img, 1 / 255.0, (64, 48), swapRB: true), view, "blobFromImage"));
        }

        /// <summary>
        ///     Reversed image views big enough for the OpenCV permutation route (<see cref="ReversedLayoutCopy"/>), in every
        ///     depth: the channel order with 2, 3 and 4 channels, rows, columns and both. The Mat must hold exactly the
        ///     bytes of NumSharp's C-ordered copy (random bit patterns, so float NaN payloads, infinities, subnormals and
        ///     negative zeros are among them), and the route must have engaged; two permutations at once stay on
        ///     NumSharp's gather.
        /// </summary>
        [TestMethod]
        public void reversed_views_copy_through_opencv_byte_for_byte()
        {
            foreach (var name in new[] { "uint8", "int8", "uint16", "int16", "int32", "float16", "float32", "float64" })
            {
                var dtype = np.dtype(name);
                foreach (int channels in new[] { 2, 3, 4 })
                {
                    using var image = RandomBits(dtype, new Shape(48, 64, channels));
                    CheckReversedCopy(image["..., ::-1"], engages: true, $"{name} x{channels} [..., ::-1]");
                    CheckReversedCopy(image["::-1"], engages: true, $"{name} x{channels} [::-1]");
                    CheckReversedCopy(image[":, ::-1"], engages: true, $"{name} x{channels} [:, ::-1]");
                    CheckReversedCopy(image["::-1, ::-1"], engages: true, $"{name} x{channels} [::-1, ::-1]");
                    CheckReversedCopy(image["::-1, :, ::-1"], engages: false, $"{name} x{channels} [::-1, :, ::-1]");
                }

                using var plane = RandomBits(dtype, new Shape(96, 64));
                CheckReversedCopy(plane["::-1"], engages: true, $"{name} 2-D [::-1]");
                CheckReversedCopy(plane[":, ::-1"], engages: true, $"{name} 2-D [:, ::-1]");
                CheckReversedCopy(plane["::-1, ::-1"], engages: true, $"{name} 2-D [::-1, ::-1]");
                // A reversed AND strided axis is not one cv::Mat views even un-reversed: NumSharp's gather.
                CheckReversedCopy(plane[":, ::-2"], engages: false, $"{name} 2-D [:, ::-2]");
            }
        }

        /// <summary>Converts <paramref name="view"/> (Auto, so cv2's copy) and compares the Mat's bytes with NumSharp's
        ///     own C-ordered copy.</summary>
        /// <param name="view">A reversed view.</param>
        /// <param name="engages">Whether the OpenCV route must have made the copy.</param>
        /// <param name="what">The case, for messages.</param>
        private static void CheckReversedCopy(NDArray view, bool engages, string what)
        {
            long engaged = ReversedLayoutCopy.Engaged;
            using var mat = new Mat(view);
            Assert.AreEqual(engaged + (engages ? 1 : 0), ReversedLayoutCopy.Engaged, $"{what}: the OpenCV route {(engages ? "engaged" : "stayed out")}");
            using var pixels = mat.data;
            using var expected = view.copy('C');
            CollectionAssert.AreEqual(expected.shape, pixels.shape, $"{what}: shape");
            CollectionAssert.AreEqual(expected.tobytes(), pixels.tobytes(), $"{what}: the same bytes as NumSharp's copy");
        }

        /// <summary>An array of <paramref name="dtype"/> holding pseudo-random BIT PATTERNS (fixed seed), so a byte
        ///     comparison sees any value an operation would canonicalise.</summary>
        /// <param name="dtype">The element type.</param>
        /// <param name="shape">The shape.</param>
        /// <returns>A new C-contiguous array.</returns>
        private static NDArray RandomBits(DType dtype, Shape shape)
        {
            var bytes = new byte[shape.size * dtype.itemsize];
            new Random(20261109).NextBytes(bytes);
            using var flat = np.frombuffer(bytes, dtype);
            using var shaped = flat.reshape(shape);
            return shaped.copy('C');
        }

        /// <summary>Runs <c>cv2.flip(x, 1)</c> on both sides and requires the same result or the same refusal.</summary>
        /// <param name="x">The array (handed to cv2 as a numpy view with the same layout).</param>
        /// <param name="what">The case, for messages.</param>
        private static void CompareFlip(NDArray x, string what)
        {
            Mat mine = null;
            OpenCVException sharpError = null;
            try
            {
                mine = cv2.flip(x, FLIP_MODE.FLIP_HORIZONTAL_MODE);
            }
            catch (OpenCVException e)
            {
                sharpError = e;
            }

            Oracle.Run(() =>
            {
                object expected = null;
                Python.Runtime.PythonException pythonError = null;
                try
                {
                    expected = Oracle.cv2.flip(x, 1);
                }
                catch (Python.Runtime.PythonException e)
                {
                    pythonError = e;
                }

                if (pythonError is not null || sharpError is not null)
                {
                    Assert.IsNotNull(pythonError, $"{what}: cv2 accepted the array, SharpCV refused it: {sharpError?.Message}");
                    Assert.IsNotNull(sharpError, $"{what}: cv2 refused the array ({pythonError.Message.Split('\n')[0]}), SharpCV accepted it");
                    return;
                }

                Oracle.AssertSame(expected, mine.data, what);
            });
            mine?.Dispose();
        }

        /// <summary>The layouts of <see cref="layout_matrix_matches_cv2"/>.</summary>
        /// <param name="dtype">The element type.</param>
        /// <returns>(name, array) pairs; views share their base's memory.</returns>
        private static IEnumerable<(string, NDArray)> Layouts(DType dtype)
        {
            var base2 = np.arange(24).reshape(4, 6).astype(dtype);
            var base3 = np.arange(4 * 5 * 3).reshape(4, 5, 3).astype(dtype);
            yield return ("C-contiguous 2-D", base2);
            yield return ("transposed", base2.T);
            yield return ("reversed rows", base2["::-1"]);
            yield return ("reversed columns", base2[":, ::-1"]);
            yield return ("column-strided", base2[":, ::2"]);
            yield return ("row-strided", base2["::2"]);
            yield return ("ROI slice", base2["1:3, 1:5"]);
            yield return ("broadcast rows", np.broadcast_to(np.arange(6).astype(dtype), new Shape(4, 6)));
            yield return ("0-d", base2[0, 3]);
            yield return ("1-D", np.arange(10).astype(dtype));
            yield return ("1-D strided", np.arange(10).astype(dtype)["::3"]);
            yield return ("HWC", base3);
            yield return ("HWC channel-reversed", base3["..., ::-1"]);
            yield return ("HWC column-strided", base3[":, ::2"]);
            yield return ("HW1", base3[":, :, :1"]);
            yield return ("HWC ROI", base3["1:3, 1:4"]);
            yield return ("4-D", np.zeros(new Shape(2, 3, 4, 5), dtype));
            yield return ("600 channels (n-D)", np.zeros(new Shape(2, 2, 600), dtype));

            // Large enough for the OpenCV permutation route (ReversedLayoutCopy.MinBytes), compared with cv2 end to end.
            var big3 = np.arange(48 * 64 * 3).reshape(48, 64, 3).astype(dtype);
            var big2 = np.arange(96 * 64).reshape(96, 64).astype(dtype);
            yield return ("HWC channel-reversed, OpenCV route", big3["..., ::-1"]);
            yield return ("HWC rows and columns reversed, OpenCV route", big3["::-1, ::-1"]);
            yield return ("reversed rows, OpenCV route", big2["::-1"]);
            yield return ("reversed columns, OpenCV route", big2[":, ::-1"]);
        }

        /// <summary>The layouts of <see cref="output_layout_rules_match_cv2"/>, as factories so each side draws on a
        ///     fresh array.</summary>
        /// <param name="dtype">The element type.</param>
        /// <returns>(name, factory) pairs; the factory returns (base, the view drawn on).</returns>
        private static IEnumerable<(string, Func<(NDArray, NDArray)>)> OutputLayouts(DType dtype)
        {
            yield return ("C-contiguous", () => { var b = np.zeros(new Shape(4, 6, 3), dtype); return (b, b); });
            yield return ("ROI", () => { var b = np.zeros(new Shape(6, 8, 3), dtype); return (b, b["1:5, 2:7"]); });
            yield return ("row-strided", () => { var b = np.zeros(new Shape(8, 6, 3), dtype); return (b, b["::2"]); });
            yield return ("column-strided", () => { var b = np.zeros(new Shape(4, 12, 3), dtype); return (b, b[":, ::2"]); });
            yield return ("reversed rows", () => { var b = np.zeros(new Shape(4, 6, 3), dtype); return (b, b["::-1"]); });
            yield return ("channel-reversed", () => { var b = np.zeros(new Shape(4, 6, 3), dtype); return (b, b["..., ::-1"]); });
            yield return ("transposed", () => { var b = np.zeros(new Shape(6, 4), dtype); return (b, b.T); });
            yield return ("broadcast (read-only)", () => { var b = np.zeros(new Shape(6), dtype); return (b, np.broadcast_to(b, new Shape(4, 6))); });
        }
    }
}
