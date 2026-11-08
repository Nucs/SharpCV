using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using NumSharp;
using NumSharp.Backends.Unmanaged;

namespace SharpCV
{
    /// <summary>
    ///     An OpenCV <c>cv::Mat</c> whose pixels NumSharp can read and write without copying: <see cref="data"/> (or
    ///     <see cref="AsNDArray"/>) is an <see cref="NDArray"/> view over the Mat's own memory, shaped and typed as
    ///     opencv-python shapes its arrays, and <c>new Mat(nd)</c> is a Mat over an NDArray's memory, converted by
    ///     opencv-python's rules.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Lifetimes.</b> A view holds its own cv::Mat reference, so it stays valid after this Mat is disposed or
    ///     collected (NumPy's <c>base</c> keeps an array alive the same way). A Mat made from an NDArray holds a counted
    ///     NumSharp view of the array, so it stays valid after the caller disposes the array. Neither direction needs the
    ///     other object to be kept alive by hand.
    ///     </para>
    ///     <para>
    ///     <b>Disposal.</b> Dispose a Mat when done in loops: the native header and pixels otherwise wait for the
    ///     finalizer. A Mat created implicitly from an NDArray argument (<c>cv2.cvtColor(nd, ...)</c>) is such a
    ///     temporary.
    ///     </para>
    /// </remarks>
    public partial class Mat : DisposableObject
    {
        /// <summary>The number of NDArray views whose cv::Mat lease is still held (tests prove releases with it).</summary>
        private static long _liveViewLeases;

        /// <summary>The NDArray views over Mat memory that have not released their cv::Mat lease yet, process-wide. Returns
        ///     to its earlier value once those views are disposed or collected; a leak gate for the bridge.</summary>
        internal static long LiveViewLeases => Interlocked.Read(ref _liveViewLeases);

        /// <summary>
        ///     The NumSharp array whose memory this Mat views (a counted view of the caller's array or the copy made for
        ///     it), or null for OpenCV-allocated pixels. Owned by this Mat: disposed with it, never touched by the
        ///     finalizer (it has its own).
        /// </summary>
        private NDArray _source;

        /// <summary>True when <see cref="_source"/> is not writeable (a broadcast or read-only array): OpenCV may then
        ///     read this Mat but never write it through <see cref="OutputArray"/>.</summary>
        private bool _sourceReadOnly;

        /// <summary>
        ///     A zero-copy <see cref="NDArray"/> over the pixels, shaped as opencv-python shapes its arrays: the Mat's
        ///     dimensions, then the channels when there are more than one (<c>(410, 730, 3)</c> uint8 for a BGR image,
        ///     <c>(1, 3, 224, 224)</c> float32 for a dnn blob). Writes through it change the image. Null for an empty Mat
        ///     (opencv-python returns <c>None</c>).
        /// </summary>
        /// <remarks>
        ///     Every read builds a NEW view (a few native calls and one small lease), so hold the result in a loop instead
        ///     of reading <c>mat.data[i, j]</c> per pixel. The view's strides are the Mat's steps: an ROI's view keeps its
        ///     parent's row stride (it is not C-contiguous), exactly as cv2's arrays of a slice.
        /// </remarks>
        /// <exception cref="ObjectDisposedException">The Mat was disposed.</exception>
        public NDArray data => AsNDArray(MatCodecMode.View);

        /// <summary>The OpenCV dimension count read by the last <see cref="ndim"/> call.</summary>
        int _ndim;

        /// <summary>
        ///     The dimension count of <see cref="data"/>: the Mat's dimensions plus one when it has several channels (3 for
        ///     a BGR image, 2 for a grayscale one); 0 for an empty Mat.
        /// </summary>
        /// <exception cref="ObjectDisposedException">The Mat was disposed.</exception>
        public int ndim
        {
            get
            {
                NativeMethods.Check(cv2_native_api.core_Mat_dims(Handle, out _ndim));
                return _ndim + (Channels > 1 ? 1 : 0);
            }
        }

        /// <summary>
        ///     The NumSharp dtype of the pixels' depth (uint8, int8, uint16, int16, int32, float32, float64, float16),
        ///     the dtype <see cref="data"/> has.
        /// </summary>
        /// <exception cref="ObjectDisposedException">The Mat was disposed.</exception>
        public DType dtype => MatDtypes.ToDType(MatType.Depth());

        /// <summary>The type read by the last <see cref="MatType"/> call; reset whenever OpenCV may reallocate the Mat.</summary>
        MatType _matType = MatType.Unknown;

        /// <summary>The OpenCV type: depth and channel count (<see cref="MatType.CV_8UC3"/> for a BGR image).</summary>
        /// <exception cref="ObjectDisposedException">The Mat was disposed.</exception>
        public MatType MatType
        {
            get
            {
                if (_matType != MatType.Unknown)
                    return _matType;

                NativeMethods.Check(cv2_native_api.core_Mat_type(Handle, out _matType));
                return _matType;
            }
        }

        /// <summary>The OpenCV element depth (the channel-independent part of <see cref="MatType"/>).</summary>
        /// <exception cref="ObjectDisposedException">The Mat was disposed.</exception>
        public MatDepth Depth => MatType.Depth();

        /// <summary>The sizes cached by <see cref="shape"/>; reset whenever OpenCV may reallocate the Mat.</summary>
        int[] _dims;

        /// <summary>The shape cached by <see cref="shape"/>.</summary>
        Shape _shape;

        /// <summary>
        /// HWC
        /// height * width * channel
        /// </summary>
        /// <remarks>
        ///     The shape of <see cref="data"/> (a NumSharp <see cref="Shape"/>); <c>(0, 0)</c> for an empty Mat. Cached
        ///     until the Mat is next handed to OpenCV as an output, which may reallocate it.
        /// </remarks>
        /// <exception cref="ObjectDisposedException">The Mat was disposed.</exception>
        public Shape shape
        {
            get
            {
                if(_dims != null)
                    return _shape;

                var dims = new int[ndim];
                NativeMethods.Check(cv2_native_api.core_Mat_dims(Handle, out _ndim));

                for (int i = 0; i < _ndim; i++)
                    NativeMethods.Check(cv2_native_api.core_Mat_sizeAt(Handle, i, out dims[i]));

                if(Channels > 1)
                    dims[_ndim] = Channels;

                // An empty Mat has no dimensions in OpenCV (dims == 0); a 0-d NumSharp shape would claim one element.
                if (dims.Length == 0)
                    dims = new[] { 0, 0 };

                _shape = new Shape(dims);
                _dims = dims;
                return _shape;
            }
        }

        /// <summary>The number of channel values (<c>rows * cols * channels</c> for an image); 0 for an empty Mat.</summary>
        /// <exception cref="ObjectDisposedException">The Mat was disposed.</exception>
        public long size => shape.size;

        /// <summary>The <c>cv::_InputArray</c> wrapper, created on first use and freed with the Mat.</summary>
        IntPtr _intputArray;

        /// <summary>
        ///     The <c>cv::_InputArray</c> OpenCV functions read this Mat through. It points at the Mat's header: pass it
        ///     only while the Mat is alive (wrappers call <c>GC.KeepAlive(mat)</c> after the native call).
        /// </summary>
        /// <exception cref="ObjectDisposedException">The Mat was disposed.</exception>
        public IntPtr InputArray
        {
            get
            {
                if(_intputArray == IntPtr.Zero)
                {
                    var handle = Handle;
                    IntPtr ia = IntPtr.Zero;
                    NativeMethods.Call(() => cv2_native_api.core_InputArray_new_byMat(handle, out ia));
                    _intputArray = ia;
                }
                return _intputArray;
            }
        }

        /// <summary>The <c>cv::_OutputArray</c> wrapper, created on first use and freed with the Mat.</summary>
        IntPtr _outputArray;

        /// <summary>
        ///     The <c>cv::_OutputArray</c> OpenCV functions write (and may reallocate) this Mat through; also used for
        ///     in-place <c>InputOutputArray</c> parameters (drawing). Reading it resets the cached type and shape, since the
        ///     call it is passed to may change them.
        /// </summary>
        /// <exception cref="ObjectDisposedException">The Mat was disposed.</exception>
        /// <exception cref="OpenCVException">The Mat views a read-only NDArray (a broadcast view): cv2 refuses such an
        ///     array as an output argument too.</exception>
        public IntPtr OutputArray
        {
            get
            {
                if (_sourceReadOnly)
                    throw OpenCVException.BadArgument("img marked as output argument, but provided NumPy array marked as readonly");

                InvalidateCaches();
                if(_outputArray == IntPtr.Zero)
                {
                    var handle = Handle;
                    IntPtr oa = IntPtr.Zero;
                    NativeMethods.Call(() => cv2_native_api.core_OutputArray_new_byMat(handle, out oa));
                    _outputArray = oa;
                }
                return _outputArray;
            }
        }

        /// <summary>The channel count cached by <see cref="Channels"/>; -1 until read.</summary>
        int _channels = -1;

        /// <summary>The channel count (1 for grayscale, 3 for BGR, 4 for BGRA).</summary>
        /// <exception cref="ObjectDisposedException">The Mat was disposed.</exception>
        public int Channels
        {
            get
            {
                if (_channels > -1)
                    return _channels;

                NativeMethods.Check(cv2_native_api.core_Mat_channels(Handle, out _channels));
                return _channels;
            }
        }

        /// <summary>The native cv::Mat pointer.</summary>
        /// <exception cref="ObjectDisposedException">The Mat was disposed: OpenCV would dereference a null header.</exception>
        internal IntPtr Handle => _handle != IntPtr.Zero ? _handle : throw new ObjectDisposedException(nameof(Mat));

        /// <summary>
        ///     The pixels as a zero-copy NDArray view; the same as <see cref="data"/>.
        /// </summary>
        /// <returns>The view, or null for an empty Mat.</returns>
        /// <exception cref="ObjectDisposedException">The Mat was disposed.</exception>
        public unsafe NDArray GetData()
            => AsNDArray(MatCodecMode.View);

        /// <summary>
        ///     The pixels as an NDArray: a view that shares this Mat's memory (<see cref="MatCodecMode.Auto"/> and
        ///     <see cref="MatCodecMode.View"/>, which agree because every Mat layout is viewable), or an independent
        ///     C-contiguous copy (<see cref="MatCodecMode.Copy"/>).
        /// </summary>
        /// <param name="mode">Share or copy.</param>
        /// <returns>The array, or null for an empty Mat (opencv-python's <c>None</c>).</returns>
        /// <exception cref="ObjectDisposedException">The Mat was disposed.</exception>
        /// <exception cref="OpenCVException">OpenCV failed to create the view's lease.</exception>
        /// <remarks>
        ///     <para>
        ///     The Mat builds ONE view per allocation of its pixels and hands out NumSharp aliases of it (<c>view()</c>,
        ///     a counted reference): the first read costs a few native calls and a lease, every later one a NumSharp
        ///     alias (sub-microsecond). The view takes its own cv::Mat header (<c>new cv::Mat(*m)</c>, a reference on
        ///     OpenCV-allocated pixels) and hands it to NumSharp as the buffer's release hook, which deletes it when the
        ///     last NumSharp array over the buffer is disposed or collected. A Mat over NumSharp memory also lends the
        ///     view a counted reference to that memory, so views outlive both this Mat and the caller's array.
        ///     </para>
        ///     <para>
        ///     Handing the Mat to OpenCV as an output (which may reallocate it) drops the Mat's own view; arrays handed
        ///     out earlier keep the pixels they were made over, as a numpy array keeps its buffer.
        ///     </para>
        /// </remarks>
        public unsafe NDArray AsNDArray(MatCodecMode mode = MatCodecMode.Auto)
        {
            var cache = Cache;
            if (cache is null)
                return null;

            // An alias, never the cached array itself: the caller may dispose what it gets.
            NDArray result = mode == MatCodecMode.Copy ? cache.View.copy() : cache.View.view();
            GC.KeepAlive(this);
            return result;
        }

        /// <summary>
        ///     The Mat's own view of its current pixels and the layout it was built from, kept until the Mat is disposed
        ///     or handed to OpenCV as an output. Owns the view (its lease); nothing outside the Mat ever sees this
        ///     instance, so nothing outside can dispose it.
        /// </summary>
        private sealed class ViewCache : IDisposable
        {
            /// <summary>The view over the pixels (holding the cv::Mat lease).</summary>
            public readonly NDArray View;

            /// <summary>The layout the view was built from: the indexers read pixels through it without a native
            ///     call.</summary>
            public readonly MatLayout Layout;

            /// <summary>Pairs a view with its layout.</summary>
            /// <param name="view">The view; owned from here on.</param>
            /// <param name="layout">Its layout.</param>
            public ViewCache(NDArray view, MatLayout layout)
            {
                View = view;
                Layout = layout;
            }

            /// <summary>Releases the cached view's reference (aliases handed out keep their own).</summary>
            public void Dispose() => View.Dispose();
        }

        /// <summary>The current <see cref="ViewCache"/>, or null until first use / after an invalidation.</summary>
        private ViewCache _cache;

        /// <summary>
        ///     The cached view of the current pixels, built on first use (a race builds two and keeps one); null for an
        ///     empty Mat, which is not cached because it has nothing to lease.
        /// </summary>
        /// <exception cref="ObjectDisposedException">The Mat was disposed.</exception>
        /// <exception cref="OpenCVException">OpenCV failed to create the view's lease.</exception>
        private ViewCache Cache
        {
            get
            {
                var cache = Volatile.Read(ref _cache);
                if (cache is not null)
                    return cache;

                var layout = MatLayout.Read(Handle);
                if (layout.IsEmpty)
                    return null;

                // A Mat with a zero-size axis has data but nothing to view: a fresh empty array needs no lease.
                NDArray view = layout.ElementCount == 0
                    ? np.empty(new Shape(layout.Shape), layout.DType)
                    : LeaseView(layout);
                var fresh = new ViewCache(view, layout);
                var prior = Interlocked.CompareExchange(ref _cache, fresh, null);
                if (prior is null)
                    return fresh;

                // Another thread published first: keep its view and release this one.
                fresh.Dispose();
                return prior;
            }
        }

        /// <summary>
        ///     Builds the NDArray view over the pixels <paramref name="layout"/> describes, with its own cv::Mat lease as
        ///     the NumSharp buffer's release hook.
        /// </summary>
        /// <param name="layout">The Mat's current layout (non-empty).</param>
        /// <returns>The view.</returns>
        /// <exception cref="OpenCVException">OpenCV failed to create the lease.</exception>
        private unsafe NDArray LeaseView(in MatLayout layout)
        {
            var handle = Handle;
            IntPtr lease = IntPtr.Zero;
            NativeMethods.Call(() => cv2_native_api.core_Mat_new12(handle, out lease));

            // A Mat over NumSharp memory: the lease header does not own those pixels, so the view also keeps a counted
            // NumSharp reference to them.
            NDArray keep = _source?.view();
            Interlocked.Increment(ref _liveViewLeases);
            int released = 0;

            void Release()
            {
                // NumSharp calls this once when the last array over the buffer is disposed, or from the buffer's
                // finalizer; the flag also covers the failure path below.
                if (Interlocked.Exchange(ref released, 1) != 0)
                    return;
                try
                {
                    cv2_native_api.core_Mat_delete(lease);
                }
                catch (SEHException)
                {
                    // Deleting a header cannot fail; never let a native surprise escape a finalizer.
                }

                // `keep` is NOT disposed here: on the finalizer path it may have been finalized already, and a Dispose after
                // its finalizer would release its reference twice. Referencing it roots it until this closure dies; its own
                // finalizer then drops the reference.
                GC.KeepAlive(keep);
                Interlocked.Decrement(ref _liveViewLeases);
            }

            var bytes = new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(layout.Data, layout.ByteExtent, Release));
            try
            {
                return np.ndarray(new Shape(layout.Shape), layout.DType, buffer: bytes, offset: 0, strides: layout.ByteStrides);
            }
            catch
            {
                Release();
                throw;
            }
        }

        /// <summary>Kept from SharpCV's first NumSharp version, where a closure over it rooted the Mat for the life of a
        ///     view; views now hold their own cv::Mat lease instead.</summary>
        /// <param name="ptr">Ignored.</param>
        [MethodImpl(MethodImplOptions.NoOptimization)]
        private void DoNothing(IntPtr ptr)
        {
            var p = ptr;
        }

        /// <summary>The pixels converted to <paramref name="dtype"/>, as a new array (NumPy's <c>astype</c>, unsafe
        ///     casting: values out of the target's range wrap).</summary>
        /// <param name="dtype">The target dtype.</param>
        /// <returns>A new array, or null for an empty Mat.</returns>
        /// <exception cref="ObjectDisposedException">The Mat was disposed.</exception>
        public NDArray astype(DType dtype)
        {
            using var view = AsNDArray(MatCodecMode.View);
            return view?.astype(dtype);
        }

        /// <summary>The pixels converted to <typeparamref name="T"/>'s dtype, as a new array.</summary>
        /// <typeparam name="T">The element type (<c>float</c> for float32, ...).</typeparam>
        /// <returns>A new array, or null for an empty Mat.</returns>
        /// <exception cref="ObjectDisposedException">The Mat was disposed.</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is not a NumSharp dtype.</exception>
        public NDArray astype<T>()
            => astype(np.dtype(typeof(T)));

        /// <summary>
        ///     Frees the argument wrappers and the cv::Mat header (releasing its reference on OpenCV-allocated pixels).
        ///     Runs on Dispose and on finalization, so it touches only native state.
        /// </summary>
        protected override void FreeHandle()
        {
            // Native deletes cannot fail; a surprise must not escape a finalizer, so statuses and SEH are both ignored.
            try
            {
                if (_intputArray != IntPtr.Zero)
                    cv2_native_api.core_InputArray_delete(_intputArray);
                if (_outputArray != IntPtr.Zero)
                    cv2_native_api.core_OutputArray_delete(_outputArray);
                if (_handle != IntPtr.Zero)
                    cv2_native_api.core_Mat_delete(_handle);
            }
            catch (SEHException)
            {
            }

            _intputArray = IntPtr.Zero;
            _outputArray = IntPtr.Zero;
        }

        /// <summary>
        ///     Frees the native header (base), then, on <see cref="DisposableObject.Dispose()"/> only, this Mat's reference
        ///     to the NumSharp memory it views; on finalization that array's own finalizer drops the reference.
        /// </summary>
        /// <param name="disposing">True from Dispose, false from the finalizer.</param>
        protected override void Dispose(bool disposing)
        {
            // The header goes first: it may point into _source's memory.
            base.Dispose(disposing);
            if (disposing)
            {
                // The Mat's own view holds a lease of its own; aliases handed out keep theirs.
                Interlocked.Exchange(ref _cache, null)?.Dispose();
                _source?.Dispose();
                _source = null;
            }
        }

        /// <summary>The shape and type, e.g. <c>(410, 730, 3) CV_8UC3</c>.</summary>
        /// <returns>The text.</returns>
        public override string ToString()
        {
            return $"{shape} {MatType}";
        }

        /// <summary>The single-channel Mat type that holds <paramref name="type"/>'s values.</summary>
        /// <param name="type">A CLR element type (<c>typeof(float)</c>).</param>
        /// <returns>The Mat type; int64 and uint64 give <see cref="MatType.CV_32SC1"/>, the cast cv2 applies.</returns>
        /// <exception cref="NotSupportedException">A type OpenCV has no depth for (bool, uint32, decimal, ...); this
        ///     returned <see cref="MatType.CV_8UC1"/> silently before SharpCV 0.20.</exception>
        public MatType FromType(Type type)
            => FromType(np.dtype(type).GetTypeCode());

        /// <summary>The single-channel Mat type that holds NumSharp <paramref name="typeCode"/> values.</summary>
        /// <param name="typeCode">The NumSharp storage type.</param>
        /// <returns>The Mat type; Int64 and UInt64 give <see cref="MatType.CV_32SC1"/>, the cast cv2 applies.</returns>
        /// <exception cref="NotSupportedException">A type OpenCV has no depth for.</exception>
        public MatType FromType(NPTypeCode typeCode)
        {
            if (MatDtypes.TryToDepth(typeCode, out var depth))
                return depth.MakeType(1);
            if (typeCode == NPTypeCode.Int64 || typeCode == NPTypeCode.UInt64)
                return MatType.CV_32SC1;
            throw new NotSupportedException($"OpenCV has no depth for {typeCode}.");
        }

        /// <summary>
        ///     Forgets the cached type, channel count, shape and the Mat's own view: the next read asks OpenCV again.
        ///     Called whenever the Mat is handed to OpenCV as an output, which may reallocate it with another size or type
        ///     (the old view is released; arrays handed out from it keep the old pixels alive).
        /// </summary>
        internal void InvalidateCaches()
        {
            _matType = MatType.Unknown;
            _channels = -1;
            _dims = null;
            _shape = default;
            Interlocked.Exchange(ref _cache, null)?.Dispose();
        }

        /// <summary>
        ///     Makes this Mat (a header OpenCV created over <paramref name="parent"/>'s pixels, e.g. an ROI) keep the
        ///     NumSharp memory those pixels live in, if any.
        /// </summary>
        /// <param name="parent">The Mat the header was made from.</param>
        internal void InheritSource(Mat parent)
        {
            // `is not null`: NDArray overloads != element-wise, so `!= null` would build a bool array.
            if (parent._source is not null)
            {
                _source = parent._source.view();
                _sourceReadOnly = parent._sourceReadOnly;
            }
        }

        #region ImEncode / ToBytes

        /// <summary>
        /// Encodes an image into a memory buffer.
        /// </summary>
        /// <param name="ext">The file extension that picks the format (".png", ".jpg").</param>
        /// <param name="prms">Format-specific parameters.</param>
        /// <returns>The encoded bytes.</returns>
        /// <exception cref="OpenCVException">OpenCV cannot encode the image in that format.</exception>
        public byte[] ImEncode(string ext = ".png", int[]? prms = null)
        {
            //ThrowIfDisposed();
            SharpCV.cv_api.ImEncode(ext, this, out var buf, prms);
            return buf;
        }

        ///// <summary>
        ///// Encodes an image into a memory buffer.
        ///// </summary>
        ///// <param name="ext">Encodes an image into a memory buffer.</param>
        ///// <param name="prms">Format-specific parameters.</param>
        ///// <returns></returns>
        //public byte[] ImEncode(string ext = ".png", params ImageEncodingParam[] prms)
        //{
        //    //ThrowIfDisposed();
        //    SharpCV.cv_api.ImEncode(ext, this, out var buf, prms);
        //    return buf;
        //}

        #endregion

        #region Static Initializers

        /// <summary>
        /// Creates the Mat instance from System.IO.Stream
        /// </summary>
        /// <param name="stream">The encoded image; read to its end.</param>
        /// <param name="mode">The decode flags.</param>
        /// <returns>The decoded image.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="stream"/> is null.</exception>
        /// <exception cref="ArgumentException">The stream is longer than 2 GB.</exception>
        public static Mat FromStream(System.IO.Stream stream, IMREAD_COLOR mode = IMREAD_COLOR.IMREAD_COLOR)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));
            if (stream.Length > int.MaxValue)
                throw new ArgumentException("Not supported stream (too long)");

            using var memoryStream = new System.IO.MemoryStream();
            stream.CopyTo(memoryStream);

            return FromImageData(memoryStream.ToArray(), mode);
        }

        /// <summary>
        /// Creates the Mat instance from image data (using cv::decode)
        /// </summary>
        /// <param name="imageBytes">The encoded image.</param>
        /// <param name="mode">The decode flags.</param>
        /// <returns>The decoded image (empty when the bytes are no image).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="imageBytes"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="imageBytes"/> is empty.</exception>
        public static Mat ImDecode(byte[] imageBytes, IMREAD_COLOR mode = IMREAD_COLOR.IMREAD_COLOR)
        {
            if (imageBytes == null)
                throw new ArgumentNullException(nameof(imageBytes));
            return cv_api.ImDecode(imageBytes, mode);
        }

        /// <summary>
        /// Creates the Mat instance from image data (using cv::decode)
        /// </summary>
        /// <param name="imageBytes">The encoded image.</param>
        /// <param name="mode">The decode flags.</param>
        /// <returns>The decoded image.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="imageBytes"/> is null.</exception>
        public static Mat FromImageData(byte[] imageBytes, IMREAD_COLOR mode = IMREAD_COLOR.IMREAD_COLOR)
        {
            return ImDecode(imageBytes, mode);
        }

        #endregion


    }
}
