using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace SharpCV
{
    public partial class cv_api
    {
        /// <summary>The <c>cv2.dnn</c> module: network loading and <c>blobFromImage</c>.</summary>
        public Dnn dnn { get; } = new Dnn();
    }
}
