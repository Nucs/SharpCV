#if !NET9_0_OR_GREATER
namespace System.Runtime.CompilerServices
{
    /// <summary>
    ///     The C# 13 overload-priority attribute for the net8.0 build (the type ships in the BCL from .NET 9 on). SharpCV
    ///     marks its pre-0.15 <c>Mat</c>-typed compatibility overloads with priority -1, so a C# 13+ compiler (the
    ///     default from net9.0 on, or <c>LangVersion</c> 13+) binds Mat and NDArray arguments to the
    ///     <see cref="SharpCV.InputArray"/> / <see cref="SharpCV.InputOutputArray"/> overloads instead.
    /// </summary>
    /// <remarks>
    ///     The compiler recognizes the attribute by its full name, so this internal copy is read from SharpCV's metadata
    ///     by consumers too. A C# 12 compiler (the default for net8.0 projects) ignores it: it binds an NDArray argument
    ///     to the <c>Mat</c> overload, the better conversion target, which converts through
    ///     <c>implicit operator Mat(NDArray)</c> and leaves that temporary to the finalizer, as SharpCV did before 0.15.
    /// </remarks>
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Constructor | AttributeTargets.Property, AllowMultiple = false, Inherited = false)]
    internal sealed class OverloadResolutionPriorityAttribute : Attribute
    {
        /// <summary>Gives the member a priority; among applicable overloads of one type, lower priorities are dropped
        ///     before the usual better-member rules run.</summary>
        /// <param name="priority">The priority; members default to 0.</param>
        public OverloadResolutionPriorityAttribute(int priority)
        {
            Priority = priority;
        }

        /// <summary>The priority given to the member.</summary>
        public int Priority { get; }
    }
}
#endif
