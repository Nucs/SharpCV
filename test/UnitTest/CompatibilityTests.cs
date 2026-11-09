using System;
using System.Linq;
using System.Reflection;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using SharpCV;
using static SharpCV.Binding;

namespace UnitTest
{
    /// <summary>
    ///     The pre-0.15 <c>Mat</c> / <c>NDArray</c> overloads stay: each has an <see cref="InputArray"/> /
    ///     <see cref="InputOutputArray"/> twin it forwards to, and reached the way a C# 12 caller or a method group reaches
    ///     it, it computes what the twin computes. (This project compiles with C# 13+, so its plain calls bind to the
    ///     twins: <c>[OverloadResolutionPriority(-1)]</c> on every overload here.)
    /// </summary>
    [TestClass]
    public class CompatibilityTests : Test
    {
        /// <summary>The attribute's full name: the net8.0 build carries SharpCV's internal copy, net10.0 the BCL's.</summary>
        private const string PriorityAttribute = "System.Runtime.CompilerServices.OverloadResolutionPriorityAttribute";

        /// <summary>
        ///     Every overload at priority -1 has a twin of the same name and parameter names, at the default priority, with
        ///     proxy-typed image parameters; and there are exactly the 46 Mat / NDArray signatures of the surface before
        ///     the proxies (d6ad307), so none was dropped.
        /// </summary>
        [TestMethod]
        public void every_compat_overload_has_a_proxy_twin()
        {
            int count = 0;
            foreach (var type in new[] { typeof(cv_api), typeof(Dnn), typeof(Net), typeof(Moments) })
            {
                var members = type.GetMembers(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly)
                    .OfType<MethodBase>().ToArray();
                foreach (var compat in members.Where(m => Priority(m) == -1))
                {
                    count++;
                    var names = compat.GetParameters().Select(p => p.Name).ToArray();
                    var twins = members.Where(m => m != compat && m.Name == compat.Name && Priority(m) is null
                        && m.GetParameters().Select(p => p.Name).SequenceEqual(names)
                        && m.GetParameters().Any(p => IsProxy(p.ParameterType))).ToArray();
                    Assert.AreEqual(1, twins.Length, $"{type.Name}.{compat}: its InputArray / InputOutputArray twin");
                }
            }

            Assert.AreEqual(46, count, "the pre-0.15 Mat / NDArray overloads");
        }

        /// <summary>
        ///     Invoked directly (the binding a C# 12 caller, an <c>IntPtr</c> argument or an assembly compiled against the
        ///     old surface gets; this C# 13+ project reaches them by reflection, since its own calls bind the proxies):
        ///     the same pixels as the proxies, an NDArray drawn into in place, a <c>Mat[]</c> variable concatenated, the
        ///     Moments constructor's fields.
        /// </summary>
        [TestMethod]
        public void compat_overloads_compute_what_the_proxies_compute()
        {
            using var img = cv2.imread(imgSolar);

            using var grayCompat = (Mat)Compat(nameof(cv_api.cvtColor), img, ColorConversionCodes.COLOR_BGR2GRAY, 0);
            using var gray = cv2.cvtColor(img, ColorConversionCodes.COLOR_BGR2GRAY);
            Assert.IsTrue(np.array_equal(gray.data, grayCompat.data), "cvtColor");

            using var matrix = cv2.getRotationMatrix2D((100, 80), 30, 0.75);
            using var warpedCompat = (Mat)Compat(nameof(cv_api.warpAffine), img, matrix, new Size(320, 240),
                InterpolationFlags.INTER_LINEAR, BorderTypes.BORDER_CONSTANT, null);
            using var warped = cv2.warpAffine(img, matrix, new Size(320, 240));
            Assert.IsTrue(np.array_equal(warped.data, warpedCompat.data), "warpAffine");

            using var kernel = cv2.getStructuringElement(MorphShapes.MORPH_CROSS, new Size(5, 5));
            using var openedCompat = (Mat)Compat(nameof(cv_api.morphologyEx), gray, MorphTypes.MORPH_OPEN, kernel, null, 2,
                BorderTypes.BORDER_CONSTANT, null);
            using var opened = cv2.morphologyEx(gray, MorphTypes.MORPH_OPEN, kernel, iterations: 2);
            Assert.IsTrue(np.array_equal(opened.data, openedCompat.data), "morphologyEx");

            // The NDArray drawing overload draws into the caller's array itself.
            var canvas = np.zeros(new Shape(8, 8), np.uint8);
            Compat(nameof(cv_api.rectangle), canvas, new Point(1, 1), new Point(3, 3), (Scalar)9, -1, LineTypes.LINE_8, 0);
            Assert.AreEqual(9, (byte)canvas[2, 2], "rectangle(NDArray) drew in place");

            // A Mat[] variable binds the params Mat[] overload in any C# version.
            Mat[] pair = { img, img };
            using var wide = cv2.hconcat(pair);
            using var wideProxy = cv2.hconcat(img, img);
            Assert.IsTrue(np.array_equal(wideProxy.data, wide.data), "hconcat(Mat[])");

            var compatCtor = typeof(Moments).GetConstructor(new[] { typeof(Mat), typeof(bool) });
            Assert.IsNotNull(compatCtor, "Moments(Mat, bool)");
            var compatMoments = (Moments)compatCtor.Invoke(new object[] { gray, false });
            var moments = new Moments(gray);
            Assert.AreEqual(moments.M00, compatMoments.M00, "m00");
            Assert.AreEqual(moments.Nu03, compatMoments.Nu03, "nu03");
        }

        /// <summary>Invokes the priority -1 overload of a <c>cv2</c> function whose parameter types match the arguments'
        ///     runtime types (null matches a nullable parameter).</summary>
        /// <param name="name">The function name.</param>
        /// <param name="args">The arguments, every optional parameter given.</param>
        /// <returns>The function's result.</returns>
        private static object Compat(string name, params object[] args)
        {
            var method = typeof(cv_api).GetMethods(BindingFlags.Public | BindingFlags.Instance)
                .Single(m => m.Name == name && Priority(m) == -1 && m.GetParameters().Length == args.Length
                             && m.GetParameters().Zip(args, (p, a) => a is null
                                 ? Nullable.GetUnderlyingType(p.ParameterType) is not null || !p.ParameterType.IsValueType
                                 : p.ParameterType.IsInstanceOfType(a)).All(ok => ok));
            return method.Invoke(cv2, args);
        }

        /// <summary>The overload priority a member declares, or null for the default.</summary>
        /// <param name="member">A method or constructor.</param>
        /// <returns>The priority.</returns>
        private static int? Priority(MethodBase member)
            => (int?)member.GetCustomAttributesData()
                .FirstOrDefault(a => a.AttributeType.FullName == PriorityAttribute)?.ConstructorArguments[0].Value;

        /// <summary>Whether a parameter type is one of the proxies (or an array of them, for hconcat / vconcat).</summary>
        /// <param name="type">The parameter type.</param>
        /// <returns>True for the proxies.</returns>
        private static bool IsProxy(Type type)
            => type == typeof(InputArray) || type == typeof(InputOutputArray) || type == typeof(InputArray[]);
    }
}
