using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace SharpCV
{
    /// <summary>
    ///     The base of every SharpCV object that owns a native OpenCV object: the standard dispose pattern with a finalizer
    ///     as the safety net. <see cref="FreeHandle"/> releases native state on both paths; a subclass that owns managed
    ///     disposables overrides <see cref="Dispose(bool)"/> and releases them only when <c>disposing</c> is true,
    ///     because a finalizer must not touch other finalizable objects.
    /// </summary>
    /// <remarks>
    ///     Until SharpCV 0.20 the default <see cref="FreeHandle"/> freed the handle with <c>Marshal.FreeHGlobal</c>, which
    ///     is wrong for every C++ object (it corrupted the heap for <see cref="VideoCapture"/>, which relied on it);
    ///     the method is abstract now, so every subclass states how its handle dies.
    /// </remarks>
    public abstract class DisposableObject : IDisposable
    {
        /// <summary>The native object; <see cref="IntPtr.Zero"/> once released.</summary>
        protected IntPtr _handle;

        // Flag: Has Dispose already been called?
        bool disposed = false;

        /// <summary>True once the object was disposed (or finalized); its handle is gone.</summary>
        public bool IsDisposed => disposed;

        // Public implementation of Dispose pattern callable by consumers.
        /// <summary>
        ///     Releases the native object now instead of at finalization, and the managed references it holds.
        ///     Calling it again does nothing.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        // Protected implementation of Dispose pattern.
        /// <summary>
        ///     The dispose pattern's worker: frees the native object once. An override that owns managed disposables
        ///     calls this first (the native object may point into their memory), then disposes them when
        ///     <paramref name="disposing"/> is true.
        /// </summary>
        /// <param name="disposing">True from <see cref="Dispose()"/>; false from the finalizer, where managed references
        ///     may already be finalized and must not be used.</param>
        protected virtual void Dispose(bool disposing)
        {
            if (disposed)
                return;

            if (disposing)
            {
                // Free any other managed objects here.
                //
            }

            // Free any unmanaged objects here.
            FreeHandle();

            _handle = IntPtr.Zero;

            disposed = true;
        }

        /// <summary>
        ///     Releases the native object (and any native helper objects). Runs on <see cref="Dispose()"/> AND on
        ///     finalization, possibly on the finalizer thread, so it may only touch native state and must not throw.
        /// </summary>
        protected abstract void FreeHandle();

        /// <summary>The finalizer: releases the native object when <see cref="Dispose()"/> was never called.</summary>
        ~DisposableObject()
        {
            Dispose(false);
        }
    }
}
