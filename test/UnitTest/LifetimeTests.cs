using System;
using System.Runtime.CompilerServices;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using SharpCV;
using static SharpCV.Binding;

namespace UnitTest
{
    /// <summary>
    ///     Who keeps the pixels alive: a view outlives its Mat, a Mat outlives the NDArray it views, an ROI outlives
    ///     both, and every view's cv::Mat lease is released once the view is disposed or collected (a leak gate on
    ///     <see cref="Mat.LiveViewLeases"/>).
    /// </summary>
    /// <remarks>
    ///     The process-wide lease count is shared with every other test running in parallel, so the gates poll until the
    ///     count is back at or below the baseline instead of reading it once.
    /// </remarks>
    [TestClass]
    [DoNotParallelize]
    public class LifetimeTests : Test
    {
        /// <summary>A view keeps reading the right pixels after its Mat is disposed and collected: the view holds its own
        ///     cv::Mat reference.</summary>
        [TestMethod]
        public void view_outlives_mat()
        {
            var mat = cv2.imread(imgSolar);
            var view = mat.data;
            var snapshot = view.copy();
            mat.Dispose();
            Collect();
            // Fresh allocations that would reuse freed memory if the view had not kept it.
            for (int i = 0; i < 4; i++)
                using (var other = cv2.imread(imgSolar)) { }
            Assert.IsTrue(np.array_equal(snapshot, view), "the view still reads the image");
            view.Dispose();
        }

        /// <summary>A Mat keeps reading the right pixels after the caller disposes the NDArray it views (it holds a
        ///     counted NumSharp view).</summary>
        [TestMethod]
        public void mat_outlives_ndarray()
        {
            var nd = np.arange(4 * 6).reshape(4, 6).astype(np.uint8);
            var expected = nd.copy();
            var mat = new Mat(nd, MatCodecMode.View);
            nd.Dispose();
            Collect();
            Assert.IsTrue(np.array_equal(expected, mat.data));
            var (ok, png) = cv2.imencode(".png", mat);
            Assert.IsTrue(ok);
            Assert.IsTrue(np.array_equal(expected, cv2.imdecode(png, IMREAD_COLOR.IMREAD_GRAYSCALE).data));
            mat.Dispose();
        }

        /// <summary>An ROI of a Mat over an NDArray keeps the NDArray's memory after both the parent Mat and the caller's
        ///     array are gone, and so does a view of that ROI.</summary>
        [TestMethod]
        public void roi_of_ndarray_mat_outlives_everything()
        {
            var nd = np.arange(6 * 8 * 3).reshape(6, 8, 3).astype(np.uint8);
            var expected = nd["1:4, 2:6"].copy();
            var mat = new Mat(nd, MatCodecMode.View);
            var roi = mat[(1, 4), (2, 6)];
            mat.Dispose();
            nd.Dispose();
            Collect();
            var view = roi.data;
            roi.Dispose();
            Collect();
            Assert.IsTrue(np.array_equal(expected, view));
        }

        /// <summary>
        ///     All views of one allocation share ONE cv::Mat lease (the Mat's own view; each <c>mat.data</c> is a NumSharp
        ///     alias of it), and the lease is released once the Mat and every alias are disposed, in either order.
        /// </summary>
        [TestMethod]
        public void disposed_views_release_their_leases()
        {
            long baseline = Mat.LiveViewLeases;
            var mat = cv2.imread(imgSolar);
            var views = new NDArray[50];
            for (int i = 0; i < views.Length; i++)
                views[i] = mat.data;
            Assert.AreEqual(baseline + 1, Mat.LiveViewLeases, "one lease for the allocation, not one per view");

            // The Mat goes first: the aliases still hold the lease.
            mat.Dispose();
            Assert.AreEqual(baseline + 1, Mat.LiveViewLeases, "aliases keep the lease after the Mat is disposed");
            foreach (var v in views)
                v.Dispose();
            Assert.IsTrue(WaitFor(() => Mat.LiveViewLeases <= baseline), $"lease released with the last alias: {Mat.LiveViewLeases} live, baseline {baseline}");
        }

        /// <summary>
        ///     Handing a Mat to OpenCV as an output drops its own view, because OpenCV may reallocate the pixels: after a
        ///     reallocation a view taken before keeps the old pixels and the next view sees the new ones; when OpenCV
        ///     writes into the existing buffer instead (a frame of the same size), the earlier view sees the write, as
        ///     a numpy array does.
        /// </summary>
        [TestMethod]
        public void output_use_refreshes_the_view()
        {
            using var capture = cv2.VideoCapture(mp4Road);
            using var frame = new Mat(np.zeros(new Shape(10, 10, 3), np.uint8));
            var small = frame.data;
            capture.read(frame);
            var big = frame.data;
            CollectionAssert.AreEqual(new long[] { 10, 10, 3 }, small.shape, "the old view keeps the old pixels");
            Assert.AreEqual(0L, (long)np.count_nonzero(small));
            CollectionAssert.AreEqual(new long[] { 720, 1280, 3 }, big.shape, "the new view sees the reallocated frame");

            var before = big.copy();
            capture.read(frame);
            Assert.IsFalse(np.array_equal(before, big), "a same-size read writes through the existing view");
        }

        /// <summary>Views that are never disposed release their leases when collected, sub-views included.</summary>
        [TestMethod]
        public void collected_views_release_their_leases()
        {
            long baseline = Mat.LiveViewLeases;
            using (var mat = cv2.imread(imgSolar))
                MakeAndDropViews(mat, 50);
            Assert.IsTrue(WaitFor(() =>
            {
                Collect();
                return Mat.LiveViewLeases <= baseline;
            }), $"leases released by the finalizer: {Mat.LiveViewLeases} live, baseline {baseline}");
        }

        /// <summary>
        ///     Results that are never disposed (opencv-python style) stay bounded: the native-memory tracker collects once
        ///     the undisposed OpenCV memory climbs a budget, and the finalizers give it back. Before the tracker, 300 such
        ///     1080p results held about 600 MB and caused no collection.
        /// </summary>
        [TestMethod]
        public void undisposed_results_stay_bounded()
        {
            using var frame = cv2.resize(cv2.imread(imgSolar), (1920, 1080));
            long collections = NativeMemoryTrackerCollections();
            for (int i = 0; i < 120; i++)
            {
                var gray = cv2.cvtColor(frame, ColorConversionCodes.COLOR_BGR2GRAY);   // 2 MB each, never disposed
                GC.KeepAlive(gray);
            }

            Assert.IsTrue(NativeMemoryTrackerCollections() > collections, "the tracker collected for the undisposed results");
            Assert.IsTrue(WaitFor(() =>
            {
                Collect();
                return NativeMemoryTrackerOutstanding() < 2 * 64L * 1024 * 1024;
            }), $"outstanding native memory settles under two budgets: {NativeMemoryTrackerOutstanding() / 1048576} MB");
        }

        /// <summary>Results that ARE disposed never make the tracker collect: their reports are taken back at once.</summary>
        [TestMethod]
        public void disposed_results_never_trigger_a_collection()
        {
            using var frame = cv2.resize(cv2.imread(imgSolar), (1920, 1080));
            long collections = NativeMemoryTrackerCollections();
            long outstanding = NativeMemoryTrackerOutstanding();
            for (int i = 0; i < 120; i++)
            {
                using var gray = cv2.cvtColor(frame, ColorConversionCodes.COLOR_BGR2GRAY);
            }

            Assert.AreEqual(collections, NativeMemoryTrackerCollections(), "no collection for disposed results");
            Assert.IsTrue(NativeMemoryTrackerOutstanding() <= outstanding + 6L * 1024 * 1024, "every report was taken back");
        }

        /// <summary>The tracker's collection count.</summary>
        /// <returns>The count.</returns>
        private static long NativeMemoryTrackerCollections() => NativeMemoryTracker.Collections;

        /// <summary>The tracker's outstanding bytes.</summary>
        /// <returns>The bytes.</returns>
        private static long NativeMemoryTrackerOutstanding() => NativeMemoryTracker.Outstanding;

        /// <summary>A disposed Mat refuses further use instead of handing OpenCV a null header.</summary>
        [TestMethod]
        public void disposed_mat_throws()
        {
            var mat = cv2.imread(imgSolar);
            mat.Dispose();
            Assert.ThrowsException<ObjectDisposedException>(() => mat.data);
            Assert.ThrowsException<ObjectDisposedException>(() => cv2.cvtColor(mat, ColorConversionCodes.COLOR_BGR2GRAY));
        }

        /// <summary>Creates views (and sub-views) and drops them without disposing.</summary>
        /// <param name="mat">The image.</param>
        /// <param name="count">How many views.</param>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void MakeAndDropViews(Mat mat, int count)
        {
            for (int i = 0; i < count; i++)
            {
                var view = mat.data;
                var row = view[i];
                GC.KeepAlive(row);
            }
        }

        /// <summary>Runs full collections and finalizers until both settle.</summary>
        private static void Collect()
        {
            for (int i = 0; i < 3; i++)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }
        }

        /// <summary>Polls a condition for up to 10 s.</summary>
        /// <param name="condition">The condition.</param>
        /// <returns>Whether it became true.</returns>
        private static bool WaitFor(Func<bool> condition)
        {
            var deadline = DateTime.UtcNow.AddSeconds(10);
            while (!condition())
            {
                if (DateTime.UtcNow > deadline)
                    return condition();
                System.Threading.Thread.Sleep(20);
            }

            return true;
        }
    }
}
