using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Runtime.InteropServices;
using System.Text;

namespace SharpCV
{
    internal partial class cv2_native_api
    {
        #region uchar
        /// <summary><c>new std::vector&lt;uchar&gt;</c>.</summary>
        /// <returns>The vector; delete it with <see cref="vector_uchar_delete"/>.</returns>
        [Pure, DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern IntPtr vector_uchar_new1();

        /// <summary><c>new std::vector&lt;uchar&gt;(size)</c>: zero-filled.</summary>
        /// <param name="size">The element count.</param>
        /// <returns>The vector.</returns>
        [Pure, DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern IntPtr vector_uchar_new2(nuint size);

        /// <summary><c>new std::vector&lt;uchar&gt;(data, data + dataLength)</c>: a copy of the bytes.</summary>
        /// <param name="data">The bytes.</param>
        /// <param name="dataLength">Their count.</param>
        /// <returns>The vector.</returns>
        [Pure, DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern IntPtr vector_uchar_new3([In] byte[] data, nuint dataLength);

        /// <summary><c>vector-&gt;size()</c>.</summary>
        /// <param name="vector">The vector.</param>
        /// <returns>The element count.</returns>
        [Pure, DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern nuint vector_uchar_getSize(IntPtr vector);

        /// <summary><c>&amp;vector-&gt;at(0)</c>. Throws a C++ <c>std::out_of_range</c> across the boundary for an
        ///     EMPTY vector (the export has no wrapper), so callers check the size first.</summary>
        /// <param name="vector">A non-empty vector.</param>
        /// <returns>The first byte's address, valid until the vector changes or is deleted.</returns>
        [Pure, DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern IntPtr vector_uchar_getPointer(IntPtr vector);

        /// <summary>Copies the vector's bytes to <paramref name="dst"/>.</summary>
        /// <param name="vector">The vector.</param>
        /// <param name="dst">A buffer of at least <see cref="vector_uchar_getSize"/> bytes.</param>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void vector_uchar_copy(IntPtr vector, IntPtr dst);

        /// <summary><c>delete vector</c>.</summary>
        /// <param name="vector">The vector; never use it, or a pointer into it, again.</param>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void vector_uchar_delete(IntPtr vector);
        #endregion

        /// <summary><c>new std::vector&lt;std::vector&lt;cv::Point&gt;&gt;</c>.</summary>
        /// <returns>The vector; delete it with <see cref="vector_vector_Point_delete"/>.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr vector_vector_Point_new1();

        /// <summary>The outer size (contour count).</summary>
        /// <param name="vector">The vector.</param>
        /// <returns>The count as a pointer-sized integer (<c>size_t</c>).</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr vector_vector_Point_getSize1(IntPtr vector);

        /// <summary><c>delete vector</c>.</summary>
        /// <param name="vector">The vector.</param>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void vector_vector_Point_delete(IntPtr vector);

        /// <summary>Writes each inner vector's size (point count) to <paramref name="size"/>.</summary>
        /// <param name="vector">The vector.</param>
        /// <param name="size">A buffer of <see cref="vector_vector_Point_getSize1"/> entries.</param>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void vector_vector_Point_getSize2(IntPtr vector, [In, Out] IntPtr[] size);

        /// <summary>Copies every inner vector to the buffer its entry of <paramref name="dst"/> points at.</summary>
        /// <param name="vec">The vector.</param>
        /// <param name="dst">One pinned buffer per contour, each large enough for its points.</param>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void vector_vector_Point_copy(IntPtr vec, IntPtr[] dst);

        /// <summary><c>new std::vector&lt;cv::Vec4i&gt;</c> (a contour hierarchy).</summary>
        /// <returns>The vector; delete it with <see cref="vector_Vec4i_delete"/>.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr vector_Vec4i_new1();

        /// <summary><c>delete vector</c>.</summary>
        /// <param name="vector">The vector.</param>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void vector_Vec4i_delete(IntPtr vector);

        /// <summary><c>new std::vector&lt;cv::Mat&gt;</c>.</summary>
        /// <returns>The vector; delete it with <see cref="vector_Mat_delete"/>.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr vector_Mat_new1();

        /// <summary><c>delete vector</c>: releases the vector's references on its Mats' pixels.</summary>
        /// <param name="vector">The vector.</param>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void vector_Mat_delete(IntPtr vector);

        /// <summary><c>vector-&gt;size()</c>. The export returns <c>size_t</c>.</summary>
        /// <param name="vector">The vector.</param>
        /// <returns>The Mat count.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern nuint vector_Mat_getSize(IntPtr vector);

        /// <summary>Assigns <c>*array[i] = vector[i]</c>: each target header SHARES its element's pixels (reference
        ///     counted), so the vector can be deleted afterwards.</summary>
        /// <param name="vector">The vector.</param>
        /// <param name="array">One existing cv::Mat per element.</param>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void vector_Mat_assignToArray(IntPtr vector, IntPtr[] array);
    }
}
