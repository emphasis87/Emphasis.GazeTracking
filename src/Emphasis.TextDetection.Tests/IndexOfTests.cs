using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Text;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emphasis.TextDetection.Tests.samples;
using FluentAssertions.Extensions;
using NUnit.Framework;
using static Emphasis.TextDetection.Tests.TestHelper;

namespace Emphasis.TextDetection.Tests
{
	public class IndexOfTests
	{
		[Test]
		public void Can_use_indexes()
		{
			var sourceBitmap = Samples.sample13;

			var w = sourceBitmap.Width;
			var h = sourceBitmap.Height;

			using var src = new UMat();

			var srcMat = sourceBitmap.ToMat();
			srcMat.CopyTo(src);

			using var gray = new UMat();
			using var resized = new UMat();
			using var canny = new UMat();

			CvInvoke.CvtColor(src, gray, ColorConversion.Bgra2Gray);
			CvInvoke.Resize(gray, resized, new Size(w * 2, h * 2));
			CvInvoke.Canny(resized, canny, 100, 40);


			
			Run("samples/sample13.png");

			canny.Save("canny.png");
			Run("canny.png");
		}
	}
}
