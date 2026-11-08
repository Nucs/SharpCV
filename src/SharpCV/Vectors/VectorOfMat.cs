using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpCV
{
    /// <summary>
    ///     A native <c>std::vector&lt;cv::Mat&gt;</c>: where OpenCV writes a list of images (split's planes,
    ///     findContours' contours). <see cref="ToArray"/> hands each element out as a <see cref="Mat"/> that shares its
    ///     pixels, so the vector can be disposed right after.
    /// </summary>
    public class VectorOfMat : DisposableObject
    {
        /// <summary>An empty vector for OpenCV to fill.</summary>
        public VectorOfMat()
        {
            NativeMethods.Initialize();
            _handle = cv2_native_api.vector_Mat_new1();
        }

        /// <summary>Wraps an existing <c>std::vector&lt;cv::Mat&gt;*</c> and takes ownership of it.</summary>
        /// <param name="ptr">The vector.</param>
        public VectorOfMat(IntPtr ptr) => _handle = ptr;

        /// <summary>Deletes the vector, releasing its references on the elements' pixels (Mats handed out keep their
        ///     own).</summary>
        protected override void FreeHandle()
        {
            if (_handle != IntPtr.Zero)
                cv2_native_api.vector_Mat_delete(_handle);
        }

        /// <summary>The element count. The export returns <c>size_t</c>; it is narrowed to the int a C# array
        ///     length takes.</summary>
        public int Size => (int)cv2_native_api.vector_Mat_getSize(_handle);

        /// <summary>The elements as Mats that share their pixels with the vector's (reference counted).</summary>
        /// <returns>One Mat per element; an empty array for an empty vector.</returns>
        public Mat[] ToArray() => ToArray<Mat>();

        /// <summary>The elements as <typeparamref name="T"/> Mats that share their pixels (reference counted).</summary>
        /// <typeparam name="T">The Mat type to create per element.</typeparam>
        /// <returns>One Mat per element.</returns>
        /// <exception cref="OpenCVException">OpenCV failed to create a target header.</exception>
        public T[] ToArray<T>()
            where T : Mat, new()
        {
            var size = Size;
            if (size == 0)
                return new T[0];

            var dst = new T[size];
            var dstPtr = new IntPtr[size];
            for (var i = 0; i < size; i++)
            {
                var m = new T();
                dst[i] = m;
                dstPtr[i] = m;
            }
            cv2_native_api.vector_Mat_assignToArray(_handle, dstPtr);
            GC.KeepAlive(this);

            // Each element now shares its buffer with the vector, which is deleted next: the Mats become the owners the
            // GC must hear about (split planes are full-size images).
            foreach (var m in dst)
                m.TrackNativeMemory();

            return dst;
        }

        /// <summary>The native vector pointer, for P/Invoke arguments.</summary>
        /// <param name="mat">The vector.</param>
        /// <returns>Its handle.</returns>
        public static implicit operator IntPtr(VectorOfMat mat)
            => mat._handle;
    }
}
