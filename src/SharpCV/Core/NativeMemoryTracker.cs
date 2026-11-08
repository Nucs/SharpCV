using System;
using System.Threading;

namespace SharpCV
{
    /// <summary>
    ///     Keeps the native pixel memory that undisposed Mats hold in check. The GC sees a Mat as a ~150-byte object and
    ///     never collects for the megabytes of OpenCV memory behind it, so code written like opencv-python (results never
    ///     disposed) grew without bound: 300 undisposed 1080p <c>resize</c> results held 5.3 GB and caused no collection.
    ///     The tracker counts the bytes held by Mats that are alive (reported, and not yet given back by Dispose or the
    ///     finalizer) and runs one full collection whenever that count climbs <see cref="Budget"/> above what was alive
    ///     after the previous one, which queues the dead Mats' finalizers.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Why not <see cref="GC.AddMemoryPressure"/>.</b> The runtime counts every report towards its next induced
    ///     gen2 collection, even when the Mat is disposed a microsecond later: with pressure reports a loop that disposes
    ///     every result ran 100 gen2 collections in 300 1080p calls (+20 % time) and an undisposed resize loop one per
    ///     call. Counting OUTSTANDING bytes instead, a program that disposes never triggers a collection, and one that
    ///     leaks pays one per <see cref="Budget"/> of garbage.
    ///     </para>
    ///     <para>
    ///     <b>Kept data.</b> After a collection the next one waits until the count is a budget above what is alive at
    ///     that point: memory the program keeps (a cache of images) does not trigger a collection per call, and memory
    ///     the finalizer gives back lowers the bar again.
    ///     </para>
    /// </remarks>
    internal static class NativeMemoryTracker
    {
        /// <summary>How many bytes of undisposed Mat memory may accumulate above the last baseline before a collection
        ///     looks for dead Mats (64 MB: about ten 1080p colour frames).</summary>
        internal const long Budget = 64L * 1024 * 1024;

        /// <summary>The bytes reported by Mats that are still alive.</summary>
        private static long _outstanding;

        /// <summary>The outstanding count that triggers the next collection.</summary>
        private static long _collectAt = Budget;

        /// <summary>1 while a thread runs a collection, so concurrent reports do not stack collections.</summary>
        private static int _collecting;

        /// <summary>How many collections the tracker has run (tests and diagnostics).</summary>
        private static long _collections;

        /// <summary>The bytes reported by Mats that are still alive.</summary>
        internal static long Outstanding => Interlocked.Read(ref _outstanding);

        /// <summary>How many collections the tracker has run in this process.</summary>
        internal static long Collections => Interlocked.Read(ref _collections);

        /// <summary>
        ///     Reports native bytes a Mat now keeps alive, and runs a full collection when the outstanding count reached
        ///     the trigger.
        /// </summary>
        /// <param name="bytes">The bytes (positive).</param>
        internal static void Add(long bytes)
        {
            long now = Interlocked.Add(ref _outstanding, bytes);
            if (now < Interlocked.Read(ref _collectAt) || Interlocked.CompareExchange(ref _collecting, 1, 0) != 0)
                return;

            try
            {
                // Raise the bar before collecting: what stays alive afterwards is either garbage the finalizer is about to
                // free (Remove lowers the bar again) or data the program keeps, which must not cause a collection per call.
                Interlocked.Exchange(ref _collectAt, now + Budget);
                // A full collection: a dead Mat may have been promoted before it died. It only queues the finalizers;
                // they free the native memory on the finalizer thread right after.
                GC.Collect();
                Interlocked.Increment(ref _collections);
            }
            finally
            {
                Volatile.Write(ref _collecting, 0);
            }
        }

        /// <summary>
        ///     Takes back bytes a Mat no longer keeps alive (Dispose or finalization), and lowers the trigger to a budget
        ///     above what is still alive. Safe on the finalizer thread.
        /// </summary>
        /// <param name="bytes">The bytes reported earlier (positive).</param>
        internal static void Remove(long bytes)
        {
            long now = Interlocked.Add(ref _outstanding, -bytes);
            long target = Math.Max(Budget, now + Budget);
            long current;
            while (target < (current = Interlocked.Read(ref _collectAt)))
            {
                if (Interlocked.CompareExchange(ref _collectAt, target, current) == current)
                    break;
            }
        }
    }
}
