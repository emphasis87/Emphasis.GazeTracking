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
		public void Sobel(bool useUmat)
		{
			var sourceBitmap = Samples.sample13;

			var w = sourceBitmap.Width;
			var h = sourceBitmap.Height;

			using var src = CreateMat(useUmat);

			var srcMat = sourceBitmap.ToMat();
			srcMat.CopyTo(src);

			var n = 1000;
			var sw = new Stopwatch();

			using var blurred = CreateMat(useUmat);
			using var gray = CreateMat(useUmat);
			using var grad_x = CreateMat(useUmat);
			using var grad_y = CreateMat(useUmat);

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

			Save(grad_x, "grad_x.png");
			Save(grad_y, "grad_y.png");

			Run("samples/sample13.png");
			Run("grad_x.png");
			Run("grad_y.png");
		}

		private static IInputOutputArray CreateMat(bool useUmat) => useUmat ? (IInputOutputArray) new UMat() : new Mat();
		private static void Save(IInputOutputArray mat, string filename)
		{
			if (mat is UMat umat)
				umat.Save(filename);
			if (mat is Mat m)
				m.Save(filename);
		}

		[TestCase(false)]
		[TestCase(true)]
		public void Grayscale(bool useUmat)
		{
			var sourceBitmap = Samples.sample13;
			
			using var src = CreateMat(useUmat);

			var srcMat = sourceBitmap.ToMat();
			srcMat.CopyTo(src);

			var n = 10000;
			var sw = new Stopwatch();
			using var gray = CreateMat(useUmat);

			CvInvoke.CvtColor(src, gray, ColorConversion.Bgra2Gray);
			sw.Start();
			for (var i = 0; i < n; i++)
			{
				CvInvoke.CvtColor(src, gray, ColorConversion.Bgra2Gray);
			}
			sw.Stop();
			
			Console.WriteLine($"{(int)(sw.Elapsed.TotalMicroseconds() / n)} us");

			Save(gray, "gray.png");
			Run("gray.png");
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
			
			using var src = CreateMat(useUmat);

			var srcMat = sourceBitmap.ToMat();
			srcMat.CopyTo(src);

			using var gray = CreateMat(useUmat);
			using var resized = CreateMat(useUmat);
			using var canny = CreateMat(useUmat);

			CvInvoke.CvtColor(src, gray, ColorConversion.Bgra2Gray);
			CvInvoke.Resize(gray, resized, new Size(w * 2, h * 2));
			CvInvoke.Canny(resized, canny, 100, 40);
			
			var n = 2000;
			var sw = new Stopwatch();
			sw.Start();

			var exCount = 0;
			for (var i = 0; i < n; i++)
			{
				CvInvoke.Canny(resized, canny, 100, 40);
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
