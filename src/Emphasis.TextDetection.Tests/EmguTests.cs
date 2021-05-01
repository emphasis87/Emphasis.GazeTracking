using System;
using System.Diagnostics;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Util;
using Emphasis.TextDetection.Tests.samples;
using FluentAssertions.Extensions;
using NUnit.Framework;
using static Emphasis.TextDetection.Tests.TestHelper;

namespace Emphasis.TextDetection.Tests
{
	public class EmguTests
	{
		[TestCase(false)]
		[TestCase(true)]
		public void Sobel(bool umat)
		{
			var sourceBitmap = Samples.sample13;

			var w = sourceBitmap.Width;
			var h = sourceBitmap.Height;

			using var src = new UMat();

			var srcMat = sourceBitmap.ToMat();
			srcMat.CopyTo(src);

			var n = 1000;
			var sw = new Stopwatch();
			
			using var blurred = umat ? (IInputOutputArray) new UMat() : new Mat(w, h, DepthType.Cv8U, 4);
			using var gray = umat ? (IInputOutputArray)new UMat() : new Mat(w, h, DepthType.Cv8U, 1);
			using var grad_x = umat ? (IInputOutputArray)new UMat() : new Mat(w, h, DepthType.Cv16S, 4);
			using var grad_y = umat ? (IInputOutputArray)new UMat() : new Mat(w, h, DepthType.Cv16S, 4);

			CvInvoke.GaussianBlur(src, blurred, new Size(3, 3), 0, 0, BorderType.Default);
			CvInvoke.CvtColor(blurred, gray, ColorConversion.Bgra2Gray);
			CvInvoke.Sobel(gray, grad_x, DepthType.Cv16S, 1, 0);
			CvInvoke.Sobel(gray, grad_y, DepthType.Cv16S, 0, 1);

			sw.Start();
			for (var i = 0; i < n; i++)
			{
				CvInvoke.GaussianBlur(src, blurred, new Size(3, 3), 0, 0, BorderType.Default);
				CvInvoke.CvtColor(blurred, gray, ColorConversion.Bgra2Gray);
				CvInvoke.Sobel(gray, grad_x, DepthType.Cv16S, 1, 0);
				CvInvoke.Sobel(gray, grad_y, DepthType.Cv16S, 0, 1);
			}

			sw.Stop();
			Console.WriteLine($"{(int)(sw.Elapsed.TotalMicroseconds() / n)} us");

			if (grad_x is UMat ux)
				ux.Save("grad_x.png");
			if (grad_x is Mat mx)
				mx.Save("grad_x.png");

			if (grad_y is UMat uy)
				uy.Save("grad_y.png");
			if (grad_y is Mat my)
				my.Save("grad_y.png");

			Run("samples/sample13.png");
			Run("grad_x.png");
			Run("grad_y.png");
		}

		[Test]
		public void Grayscale_UMat()
		{
			var sourceBitmap = Samples.sample13;
			
			using var src = new UMat();

			var srcMat = sourceBitmap.ToMat();
			srcMat.CopyTo(src);

			var n = 10000;
			var sw = new Stopwatch();
			using var gray = new UMat();

			CvInvoke.CvtColor(src, gray, ColorConversion.Bgra2Gray);
			sw.Start();
			for (var i = 0; i < n; i++)
			{
				CvInvoke.CvtColor(src, gray, ColorConversion.Bgra2Gray);
			}

			sw.Stop();
			Console.WriteLine($"{(int)(sw.Elapsed.TotalMicroseconds() / n)} us");
		}

		[Test]
		public void Grayscale_Mat()
		{
			var sourceBitmap = Samples.sample13;
			
			using var src = new Mat();

			var srcMat = sourceBitmap.ToMat();
			srcMat.CopyTo(src);

			var n = 1000;
			var sw = new Stopwatch();
			using var gray = new Mat();

			CvInvoke.CvtColor(src, gray, ColorConversion.Bgra2Gray);
			sw.Start();
			for (var i = 0; i < n; i++)
			{
				CvInvoke.CvtColor(src, gray, ColorConversion.Bgra2Gray);
			}

			sw.Stop();
			Console.WriteLine($"{(int)(sw.Elapsed.TotalMicroseconds() / n)} us");
		}

		[Test]
		public void Canny()
		{
			var sourceBitmap = Samples.sample13;

			var w = sourceBitmap.Width;
			var h = sourceBitmap.Height;

			using var src = new UMat();

			var srcMat = sourceBitmap.ToMat();
			srcMat.CopyTo(src);

			using var dest1 = new UMat();
			using var dest2 = new UMat();
			using var gray = new UMat();

			CvInvoke.CvtColor(src, gray, ColorConversion.Bgra2Gray);
			CvInvoke.Canny(gray, dest1, 100, 40);
			
			var n = 1000;
			var sw = new Stopwatch();
			sw.Start();

			var exCount = 0;
			for (var i = 0; i < n; i++)
			{
				try
				{
					CvInvoke.Canny(gray, dest1, 100, 40);
				}
				catch (Exception ex)
				{
					exCount++;
				}
			}

			sw.Stop();

			Console.WriteLine($"{(int)(sw.Elapsed.TotalMicroseconds() / n)} us");
			Console.WriteLine($"Exceptions {exCount}");

			Run("samples/sample13.png");
			//dest1.Bytes.RunAs(w, h, 1, "canny.png");
		}

		[TestCase(false)]
		[TestCase(true)]
		public void Resize_Canny(bool useUmat)
		{
			var sourceBitmap = Samples.sample13;

			var w = sourceBitmap.Width;
			var h = sourceBitmap.Height;

			IInputOutputArray CreateMat() => useUmat ? (IInputOutputArray) new UMat() : new Mat();

			void Save(IInputOutputArray mat, string filename)
			{
				if (mat is UMat umat)
					umat.Save(filename);
				if (mat is Mat m)
					m.Save(filename);
			}

			using var src = CreateMat();

			var srcMat = sourceBitmap.ToMat();
			srcMat.CopyTo(src);

			using var gray = CreateMat();
			using var resized = CreateMat();
			using var canny = CreateMat();

			CvInvoke.CvtColor(src, gray, ColorConversion.Bgra2Gray);
			CvInvoke.Resize(gray, resized, new Size(w * 2, h * 2));
			CvInvoke.Canny(resized, canny, 100, 40);

			var n = 2000;
			var sw = new Stopwatch();
			sw.Start();

			var exCount = 0;
			for (var i = 0; i < n; i++)
			{
				try
				{
					CvInvoke.Resize(gray, resized, new Size(w * 2, h * 2));
					CvInvoke.Canny(resized, canny, 100, 40);
				}
				catch (Exception ex)
				{
					exCount++;
				}
			}
			
			sw.Stop();

			Console.WriteLine($"{(int)(sw.Elapsed.TotalMicroseconds() / n)} us");
			Console.WriteLine($"Exceptions {exCount}");

			Run("samples/sample13.png");

			Save(canny, "canny.png");
			Run("canny.png");
		}

		[Test]
		public void Resize()
		{
			var sourceBitmap = Samples.sample13;

			var w = sourceBitmap.Width;
			var h = sourceBitmap.Height;

			using var src = new UMat();

			var srcMat = sourceBitmap.ToMat();
			srcMat.CopyTo(src);

			using var dest = new UMat();
			using var gray = new UMat();

			CvInvoke.CvtColor(src, gray, ColorConversion.Bgra2Gray);
			CvInvoke.Resize(gray, dest, new Size(w * 2, h * 2));

			var n = 3000;
			var sw = new Stopwatch();
			sw.Start();

			for (var i = 0; i < n; i++)
			{
				CvInvoke.Resize(gray, dest, new Size(w * 2, h * 2));
			}

			sw.Stop();

			Console.WriteLine($"{(int)(sw.Elapsed.TotalMicroseconds() / n)} us");

			Run("samples/sample13.png");
			
			dest.Save("resize.png");
			Run("resize.png");
		}

		[Test]
		public void MSER()
		{
			var sourceBitmap = Samples.sample13;
			
			var w = sourceBitmap.Width;
			var h = sourceBitmap.Height;

			using var src = new UMat();

			using var srcMat = sourceBitmap.ToMat();
			srcMat.CopyTo(src);

			using var gray = new UMat();
			
			CvInvoke.CvtColor(src, gray, ColorConversion.Bgra2Gray);
			
			using var detector = new MSERDetector(
				minArea:5, maxArea:80, edgeBlurSize: 5);
			using var msers = new VectorOfVectorOfPoint();
			using var bboxes = new VectorOfRect();

			detector.DetectRegions(gray, msers, bboxes);
			
			var sw = new Stopwatch();
			sw.Start();
			
			var n = 100;
			for(var i = 0; i < n; i++)
				detector.DetectRegions(gray, msers, bboxes);

			sw.Stop();

			Console.WriteLine($"{(int)(sw.Elapsed.TotalMicroseconds() / n)} us");

			var result = new byte[w * h];
			foreach (var mser in msers.ToArrayOfArray())
			{
				foreach (var point in mser)
				{
					result[point.Y * w + point.X] = 255;
				}
			}
			foreach (var bbox in bboxes.ToArray())
			{
				
			}


			Run("samples/sample13.png");
			//result.RunAs(w, h, 1, "mser.png");
		}
	}
}
