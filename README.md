# SharpCV

A image library combines [OpenCV](https://github.com/opencv/opencv) and [NumSharp](https://github.com/SciSharp/NumSharp) together. `SharpCV` returns `Mat` object with `NDArray` supported, which makes it easier to do data manipulation like slicing: a `Mat`'s pixels ARE an `NDArray` (no copy), and every `cv2.*` function takes an `NDArray` back, by the same rules opencv-python applies to NumPy arrays.

[![Join the chat at https://gitter.im/publiclab/publiclab](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/sci-sharp/community) [![NuGet](https://img.shields.io/nuget/dt/SharpCV.svg)](https://www.nuget.org/packages/SharpCV)

### How to use

Install SharpCV and the OpenCV runtime it binds to (SharpCV targets .NET 8 and .NET 10, like NumSharp 0.80.0):

```powershell
PM> Install-Package SharpCV
PM> Install-Package OpenCvSharp4.runtime.win -Version 4.13.0.20261107
```

SharpCV's native signatures are verified against OpenCvSharpExtern 4.13; the first call checks the loaded runtime and refuses another OpenCV version with a clear message instead of corrupting memory.

Import SharpCV, OpenCV and NumSharp
```csharp
using NumSharp;
using SharpCV;
using static SharpCV.Binding;
```

### Images are NDArrays

```csharp
NDArray img = cv2.imread("solar.jpg");        // (410, 730, 3) uint8, a view of OpenCV's own pixels
NDArray roi = img["50:150, 100:300"];          // NumSharp slicing: still a view
cv2.rectangle(roi, (10, 10), (90, 60), (0, 0, 255), -1);   // draws INTO img, as cv2 draws into a numpy array

var gray = cv2.cvtColor(img["..., ::-1"], ColorConversionCodes.COLOR_RGB2GRAY);   // any layout is accepted
var mask = (gray.data < 128).astype(np.uint8) * 255;                              // NumSharp arithmetic
var (contours, hierarchy) = cv2.findContours(mask, RetrievalModes.RETR_EXTERNAL, ContourApproximationModes.CHAIN_APPROX_SIMPLE);
```

- `mat.data` (or `mat.AsNDArray()`) is a zero-copy view, shaped as opencv-python shapes its arrays: the Mat's dimensions, then the channels when there are several (`(H, W)`, `(H, W, C)`, `(1, 3, H, W)` for a dnn blob). A region of interest keeps its parent's row stride.
- `new Mat(nd)` and every `cv2.*` image argument share the array's memory when the layout fits a `cv::Mat`, and copy it otherwise (a transposed, reversed, column-strided or broadcast array), exactly like opencv-python. `MatCodecMode.View` refuses the copy, `MatCodecMode.Copy` always copies.
- Image parameters are `InputArray` (read) or `InputOutputArray` (drawn into), like OpenCV's own C++ API: pass a `Mat` or an `NDArray`. An `NDArray` argument is converted for the one call and released when it returns, copy included, so a loop over NDArrays leaves nothing for the finalizer.
- A reversed view (`img[..., ::-1]`, the BGR / RGB idiom, or a mirrored `img[:, ::-1]`, or an upside-down `img[::-1]`) is one a `cv::Mat` cannot view, so it is copied as cv2 copies it, but by OpenCV's own permutation kernels instead of an element-by-element gather: the same bytes, and a 1080p `cv2.cvtColor(img[..., ::-1], ...)` takes 1.2 ms where opencv-python takes 10.9 ms (almost all of it NumPy's copy).
- Functions that draw INTO an image (`rectangle`, `putText`, `drawContours`) treat an `NDArray` as cv2 treats an output array: it must be writeable and viewable, otherwise `OpenCVException` ("Layout of the output array img is incompatible with cv::Mat").
- Lifetimes are automatic both ways: a view keeps its own reference to the pixels (it stays valid after the `Mat` is disposed), and a `Mat` made from an `NDArray` keeps a counted reference to the array.

### Memory

A `Mat` returned by a `cv2.*` function owns OpenCV memory the garbage collector cannot see (a 1080p frame is a small managed object over 6 MB of native pixels). Dispose results in loops (`using var gray = cv2.cvtColor(...)`) to free them at once. Code written in opencv-python style that never disposes stays bounded anyway: SharpCV counts the OpenCV memory held by live results and runs one full collection whenever it grows by 64 MB, so undisposed 1080p loops hold about 110 MB instead of growing by gigabytes. Disposed results never trigger such a collection.

| OpenCV depth | NumSharp dtype | | OpenCV depth | NumSharp dtype |
|---|---|---|---|---|
| `CV_8U` | `uint8` | | `CV_32S` | `int32` |
| `CV_8S` | `int8` | | `CV_32F` | `float32` |
| `CV_16U` | `uint16` | | `CV_64F` | `float64` |
| `CV_16S` | `int16` | | `CV_16F` | `float16` |

`int64` and `uint64` arrays are cast to `int32`, as cv2 casts them; `bool`, `uint32`, `char`, `decimal` and `complex128` arrays are refused.

Interact with `NDArray`

```csharp
NDArray kernel = new float[,]
{
    { 0, -1, 0 },
    { -1, 5, -1 },
    { 0, -1, 0 }
};

var mat = new Mat(kernel);

Assert.AreEqual((3, 3), mat.shape);
Assert.AreEqual(kernel[0], mat.data[0]); // { 0, -1, 0 }
Assert.AreEqual(kernel[1], mat.data[1]); // { -1, 5, -1 }
Assert.AreEqual(kernel[2], mat.data[2]); // { 0, -1, 0 }
```

Pixel level access

```csharp
var img = cv2.imread(imgSolar, IMREAD_COLOR.IMREAD_GRAYSCALE);
byte p = (byte)img[8, 8];
Assert.AreEqual(18, p);

img = cv2.imread(imgSolar);
var px = img[8, 8];                                  // an NDArray view of the pixel: [32, 19, 11]
var (b, g, r) = (img[8, 8, 0], img[8, 8, 1], img[8, 8, 2]);
Assert.AreEqual((32, 19, 11), (b, g, r));
img[8, 8] = np.array(new byte[] { 0, 0, 255 });     // writes the pixel
```

Convert to black and white image

```csharp
var img = cv2.imread("solar.jpg");
var gray = cv2.cvtColor(img, ColorConversionCodes.COLOR_RGB2GRAY);
var (ret, binary) = cv2.threshold(gray, 0, 255, ThresholdTypes.THRESH_BINARY | ThresholdTypes.THRESH_TRIANGLE);
cv2.imshow("black and white", binary);
cv2.waitKey(0);
```

Video capture from file or camera
```csharp
var vid = cv2.VideoCapture("road.mp4");
var (loaded, frame) = vid.read();
while (loaded)
{
    (loaded, frame) = vid.read();
    cv2.imshow("video", frame);
}
```

Decode every frame into one NumSharp buffer (no allocation per frame):
```csharp
using var vid = cv2.VideoCapture("road.mp4");
var frame = np.zeros(new Shape(720, 1280, 3), np.uint8);
while (vid.read(frame).Item1)
{
    // frame holds the current image
}
```

### Errors

OpenCV errors raise `OpenCVException`, with opencv-python's `cv2.error` message and code (`Status`):

```
OpenCV(4.13.0) ...color.simd_helpers.hpp:92: error: (-15:Bad number of channels) in function '...'
> Invalid number of channels in input image:
>     'VScn::contains(scn)'
```

### Upgrading from 0.14

- TensorFlow.NET (and its native runtime) is gone: `Mat.data` is a `NumSharp.NDArray`, `Mat.shape` a `NumSharp.Shape`, `Mat.dtype` a NumSharp `DType`, and the target frameworks are `net8.0` and `net10.0`.
- Use the OpenCvSharp4.runtime 4.13.0.20261107 package. OpenCvSharpExtern 4.13 changed export signatures (findContours now fills vectors the caller allocates).
- Defaults follow opencv-python: `getStructuringElement` anchors at the centre (MORPH_CROSS was an L-shape), `morphologyEx` uses `morphologyDefaultBorderValue()` (erosions no longer eat in from the image edge), `drawContours(Mat ...)` uses offset (0, 0), and `dnn.blobFromImage` defaults to `swapRB: false`.
- Fixed: `rotate(img, center, ...)` turned about the top-left corner (the centre was passed as ints where OpenCV reads floats); an ROI's `data` ignored the row step; `FLIP_MODE.FLIP_VERTICAL_MODE` / `FLIP_BOTH_MODE` flipped horizontally; `VideoCapture` was freed with `FreeHGlobal`; failing calls returned an empty `Mat` silently; `waitKey` returns the key.
- `img[8, 8]` returns an `NDArray` view; read a grayscale pixel with `(byte)img[8, 8]`.
- Image parameters are `InputArray` / `InputOutputArray`, and every `Mat`-typed signature of 0.14 is still there (plus the `NDArray` overloads of the drawing functions), forwarding to its proxy twin, so 0.14 call sites compile, `IntPtr` arguments included. The old overloads carry `[OverloadResolutionPriority(-1)]`: with C# 13 or later (the default from net9.0 on) `Mat` and `NDArray` arguments bind to the proxies, and an NDArray's conversion is released when the call returns. A net8.0 project defaults to C# 12, which ignores the attribute and binds an `NDArray` to the `Mat` overload, whose temporary then waits for the finalizer (kept bounded); add `<LangVersion>latest</LangVersion>` to get the proxies. One thing only C# 13+ refuses: a method group converted to a `Mat`- or `NDArray`-typed delegate (`Func<Mat, ColorConversionCodes, int, Mat> f = cv2.cvtColor;`, CS0123), because overload resolution picks the proxy first; use a lambda.

### Tests

The tests compare SharpCV with opencv-python 4.13 bit for bit, live: the same arrays cross into Python without a copy through NumSharp.Interop.pythonnet. Point `SHARPCV_PYTHON` at a Python that has `numpy` and `opencv-python==4.13.*`:

```powershell
pip install numpy opencv-python==4.13.0.92
$env:SHARPCV_PYTHON = "C:\Python312\python.exe"
dotnet test test/UnitTest
```

Without Python the comparisons report Inconclusive; with `CI=true` (or `SHARPCV_REQUIRE_PYTHON=1`) they fail instead.

If you want to learn more about the API implementation, please refer to the official [documentation](https://docs.opencv.org/).
