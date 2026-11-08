using System;

namespace SharpCV
{
    /// <summary>
    ///     A native <c>std::vector&lt;std::vector&lt;cv::Point&gt;&gt;</c>: contours as point lists, copied out to managed
    ///     <see cref="Point"/> arrays by <see cref="ToArray"/>.
    /// </summary>
    public class VectorOfVectorPoint : DisposableObject
    {
        /// <summary>An empty vector for OpenCV to fill (findContours expects the caller to allocate it).</summary>
        public VectorOfVectorPoint()
        {
            _handle = cv2_native_api.vector_vector_Point_new1();
        }

        /// <summary>Wraps an existing vector pointer and takes ownership of it.</summary>
        /// <param name="ptr">The vector.</param>
        public VectorOfVectorPoint(IntPtr ptr)
        {
            _handle = ptr;
        }

        /// <summary>Deletes the vector.</summary>
        protected override void FreeHandle()
        {
            if (_handle != IntPtr.Zero)
                cv2_native_api.vector_vector_Point_delete(_handle);
        }

        /// <summary>The number of contours (the outer size).</summary>
        public int Size1
        {
            get
            {
                var res = cv2_native_api.vector_vector_Point_getSize1(_handle).ToInt32();
                GC.KeepAlive(this);
                return res;
            }
        }

        /// <summary>The number of contours (the same as <see cref="Size1"/>).</summary>
        public int Size => Size1;

        /// <summary>The point count of each contour.</summary>
        public long[] Size2
        {
            get
            {
                var size1 = Size1;
                var size2Org = new IntPtr[size1];
                cv2_native_api.vector_vector_Point_getSize2(_handle, size2Org);
                GC.KeepAlive(this);
                var size2 = new long[size1];
                for (var i = 0; i < size1; i++)
                {
                    size2[i] = size2Org[i].ToInt64();
                }
                return size2;
            }
        }

        /// <summary>Copies every contour into a managed point array.</summary>
        /// <returns>One array per contour.</returns>
        public Point[][] ToArray()
        {
            var size1 = Size1;
            if (size1 == 0)
                return new Point[0][];
            var size2 = Size2;

            var ret = new Point[size1][];
            for (var i = 0; i < size1; i++)
            {
                ret[i] = new Point[size2[i]];
            }
            using (var retPtr = new ArrayAddress2<Point>(ret))
            {
                cv2_native_api.vector_vector_Point_copy(_handle, retPtr);
                GC.KeepAlive(this);
            }
            return ret;
        }

        /// <summary>The native vector pointer, for P/Invoke arguments (findContours fills it).</summary>
        /// <param name="vector">The vector.</param>
        /// <returns>Its handle.</returns>
        public static implicit operator IntPtr(VectorOfVectorPoint vector)
            => vector._handle;
    }
}
