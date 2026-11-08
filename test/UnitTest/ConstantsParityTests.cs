using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Python.Runtime;
using SharpCV;

namespace UnitTest
{
    /// <summary>
    ///     Every SharpCV enum member that names a cv2 constant has cv2's value: SharpCV passes the enums to OpenCV as
    ///     plain ints, so a wrong value silently selects another operation.
    /// </summary>
    [TestClass]
    public class ConstantsParityTests
    {
        /// <summary>The enums whose member names are cv2 constant names.</summary>
        private static readonly Type[] Enums =
        {
            typeof(AdaptiveThresholdTypes), typeof(BorderTypes), typeof(ColorConversionCodes), typeof(ContourApproximationModes),
            typeof(HersheyFonts), typeof(IMREAD_COLOR), typeof(InterpolationFlags), typeof(LineTypes), typeof(MorphShapes),
            typeof(MorphTypes), typeof(RetrievalModes), typeof(RotateFlags), typeof(ThresholdTypes), typeof(VideoCaptureAPIs),
            typeof(VideoCaptureProperties), typeof(WindowFlags), typeof(MatType), typeof(MatDepth),
        };

        /// <summary>cv2's names of SharpCV's <see cref="HersheyFonts"/> members.</summary>
        private static readonly Dictionary<string, string> HersheyNames = new()
        {
            ["HERSHEY_SIMPLEX"] = "FONT_HERSHEY_SIMPLEX", ["HERSHEY_PLAIN"] = "FONT_HERSHEY_PLAIN",
            ["HERSHEY_DUPLEX"] = "FONT_HERSHEY_DUPLEX", ["HERSHEY_COMPLEX"] = "FONT_HERSHEY_COMPLEX",
            ["HERSHEY_TRIPLEX"] = "FONT_HERSHEY_TRIPLEX", ["HERSHEY_COMPLEXSMALL"] = "FONT_HERSHEY_COMPLEX_SMALL",
            ["HERSHEY_SCRIPTSIMPLEX"] = "FONT_HERSHEY_SCRIPT_SIMPLEX", ["HERSHEY_SCRIPTCOMPLEX"] = "FONT_HERSHEY_SCRIPT_COMPLEX",
            ["ITALIC"] = "FONT_ITALIC",
        };

        /// <summary>Compares every member against <c>getattr(cv2, name)</c>; members cv2 does not define are listed in
        ///     the output, not failed (some are OpenCV C++-only names).</summary>
        [TestMethod]
        public void enum_values_match_cv2_constants()
        {
            var mismatches = new List<string>();
            var missing = new List<string>();
            int compared = 0;
            Oracle.Run(() =>
            {
                PyObject cv2 = Oracle.cv2;
                foreach (var type in Enums)
                {
                    foreach (var member in Enum.GetNames(type))
                    {
                        // SharpCV's HersheyFonts drop cv2's FONT_ prefix and two underscores.
                        string name = type == typeof(HersheyFonts) && HersheyNames.TryGetValue(member, out var cvName) ? cvName : member;
                        if (!cv2.HasAttr(name))
                        {
                            missing.Add($"{type.Name}.{name}");
                            continue;
                        }

                        using var constant = cv2.GetAttr(name);
                        long expected = constant.As<long>();
                        long actual = Convert.ToInt64(Enum.Parse(type, member));
                        compared++;
                        if (expected != actual)
                            mismatches.Add($"{type.Name}.{name} = {actual}, cv2.{name} = {expected}");
                    }
                }
            });

            Console.WriteLine($"compared {compared} constants; not in cv2: {string.Join(", ", missing)}");
            Assert.IsTrue(compared > 300, $"only {compared} constants were compared");
            Assert.AreEqual(0, mismatches.Count, "values differing from cv2:\n" + string.Join("\n", mismatches));
        }
    }
}
