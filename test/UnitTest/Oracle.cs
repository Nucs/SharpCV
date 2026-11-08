using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Python.Runtime;

namespace UnitTest
{
    /// <summary>
    ///     The test-side face of the opencv-python oracle: the cached <c>cv2</c> / <c>numpy</c> modules as
    ///     <c>dynamic</c> (so a parity line reads like the Python it mirrors: <c>py.cv2.cvtColor(img, code)</c>), the GIL
    ///     entry, and the bit-exact comparison every parity test ends with.
    /// </summary>
    /// <remarks>
    ///     NDArray arguments cross to Python as zero-copy numpy views and numpy results come back as NDArray views, through
    ///     the codec <see cref="PythonSession"/> registers; a Mat must be passed as its <c>.data</c> view (the codec does
    ///     not know SharpCV types).
    /// </remarks>
    internal static class Oracle
    {
        /// <summary>The cached <c>cv2</c> module: process-scoped, never disposed (disposing would drop the cache's
        ///     reference).</summary>
        private static PyObject _cv2;

        /// <summary>The cached <c>numpy</c> module.</summary>
        private static PyObject _np;

        /// <summary>opencv-python's <c>cv2</c> module; use under <see cref="Run"/> only.</summary>
        public static dynamic cv2 => _cv2 ??= Py.Import("cv2");

        /// <summary>NumPy; use under <see cref="Run"/> only.</summary>
        public static dynamic numpy => _np ??= Py.Import("numpy");

        /// <summary>The cached <c>builtins</c> module.</summary>
        private static PyObject _builtins;

        /// <summary>Python's builtins (<c>slice</c>, <c>len</c>); use under <see cref="Run"/> only.</summary>
        public static dynamic builtins => _builtins ??= Py.Import("builtins");

        /// <summary>
        ///     A Python list of numpy views of <paramref name="items"/> (cv2's list arguments: hconcat's images,
        ///     drawContours' contours). A C# array would cross as a CLR object, which cv2 rejects.
        /// </summary>
        /// <param name="items">The arrays.</param>
        /// <returns>The list; dispose it after the call (under the GIL).</returns>
        public static PyList list(params NDArray[] items)
        {
            var list = new PyList();
            foreach (var item in items)
            {
                using PyObject view = NumSharp.Interop.PythonNet.NDArrayPythonInterop.ToNumpy(item);
                list.Append(view);
            }

            return list;
        }

        /// <summary>
        ///     Runs a parity body under the GIL, or skips / fails the test when the oracle is unavailable
        ///     (<see cref="PythonSession.EnsureOrInconclusive"/>).
        /// </summary>
        /// <param name="body">The Python-touching part of the test; compute SharpCV's side before it.</param>
        /// <exception cref="AssertInconclusiveException">No oracle on a developer machine.</exception>
        public static void Run(Action body)
        {
            PythonSession.EnsureOrInconclusive();
            using (Py.GIL())
                body();
        }

        /// <summary>
        ///     <c>arr[i, j, ...]</c> in Python: pythonnet's dynamic indexer takes ONE key, so a multi-axis index is passed
        ///     as the tuple Python builds for it.
        /// </summary>
        /// <param name="arr">The Python object (a numpy array).</param>
        /// <param name="index">The keys: ints, or Python objects such as <c>builtins.slice(a, b)</c>.</param>
        /// <returns>The item or sub-array (owned by the caller).</returns>
        public static PyObject at(PyObject arr, params object[] index)
        {
            var keys = new PyObject[index.Length];
            for (int i = 0; i < index.Length; i++)
                keys[i] = index[i] is PyObject key ? key : index[i].ToPython();
            using var tuple = new PyTuple(keys);
            return arr[tuple];
        }

        /// <summary>
        ///     The NDArray of a cv2 result: a numpy array decodes through the registered codec into an NDArray view. A
        ///     method argument typed NDArray cannot take a <c>dynamic</c> cv2 result directly (the C# runtime binder does
        ///     not ask pythonnet to convert during overload resolution), so the assertions take <c>object</c> and convert
        ///     here.
        /// </summary>
        /// <param name="value">An NDArray, a Python object, or null (Python's None).</param>
        /// <returns>The NDArray, or null.</returns>
        /// <exception cref="ArgumentException"><paramref name="value"/> is neither an array nor a Python object.</exception>
        private static NDArray ToNDArray(object value) => value switch
        {
            null => null,
            NDArray nd => nd,
            PyObject py => py.IsNone() ? null : py.As<NDArray>(),
            _ => throw new ArgumentException($"not an array: {value.GetType()}"),
        };

        /// <summary>
        ///     Asserts that <paramref name="actual"/> is bit-for-bit cv2's <paramref name="expected"/>: same dtype, same
        ///     shape, every element equal (NaN equal to NaN), and reports the first differing element otherwise.
        /// </summary>
        /// <param name="expectedValue">cv2's result: a numpy array (as a Python object) or an NDArray.</param>
        /// <param name="actual">SharpCV's result.</param>
        /// <param name="what">What is compared, for the message.</param>
        /// <exception cref="AssertFailedException">The arrays differ.</exception>
        public static void AssertSame(object expectedValue, NDArray actual, string what)
        {
            NDArray expected = ToNDArray(expectedValue);
            Assert.IsNotNull(expected, $"{what}: cv2 returned None");
            Assert.IsNotNull(actual, $"{what}: SharpCV returned null");
            Assert.AreEqual(expected.dtype.name, actual.dtype.name, $"{what}: dtype");
            CollectionAssert.AreEqual(expected.shape, actual.shape, $"{what}: shape ({string.Join(", ", expected.shape)}) vs ({string.Join(", ", actual.shape)})");
            if (np.array_equal(expected, actual, equal_nan: expected.dtype.kind == 'f'))
                return;

            // Locate the first difference for the message; the comparison itself is NumSharp's. GetValue takes the
            // coordinates: a lone long[] in the indexer would be fancy indexing.
            using var diff = np.not_equal(expected, actual);
            using var where = np.argwhere(diff);
            long count = where.shape[0];
            long[] first = where[0].ToArray<long>();
            Assert.Fail($"{what}: {count} of {expected.size} elements differ; first at [{string.Join(", ", first)}]: " +
                        $"cv2 {expected.GetValue(first)} vs SharpCV {actual.GetValue(first)}");
        }

        /// <summary>
        ///     Asserts that SharpCV and cv2 both reject an input, with the same OpenCV error code when the error comes from
        ///     OpenCV itself.
        /// </summary>
        /// <param name="sharp">SharpCV's call.</param>
        /// <param name="python">cv2's call (run under the GIL by the caller's <see cref="Run"/>).</param>
        /// <param name="what">What is compared, for the message.</param>
        /// <returns>SharpCV's exception, for further checks.</returns>
        /// <exception cref="AssertFailedException">Only one side failed, or the codes differ.</exception>
        public static SharpCV.OpenCVException AssertBothFail(Action sharp, Action python, string what)
        {
            SharpCV.OpenCVException sharpError = null;
            try
            {
                sharp();
            }
            catch (SharpCV.OpenCVException e)
            {
                sharpError = e;
            }

            PythonException pythonError = null;
            try
            {
                python();
            }
            catch (PythonException e)
            {
                pythonError = e;
            }

            Assert.IsNotNull(pythonError, $"{what}: cv2 accepted the input");
            Assert.IsNotNull(sharpError, $"{what}: SharpCV accepted the input cv2 rejects ({pythonError.Message})");

            // cv2.error carries the OpenCV code in .code; argument-parser refusals (-5) carry it too.
            var value = pythonError.Value;
            if (value is not null && value.HasAttr("code"))
            {
                using var code = value.GetAttr("code");
                if (!code.IsNone())
                    Assert.AreEqual(code.As<int>(), (int)sharpError.Status, $"{what}: error code; cv2 said {pythonError.Message}");
            }

            return sharpError;
        }
    }
}
