using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using Emphasis.ComputerVision;

namespace Emphasis.TextDetection.Tests
{
	public static class TestHelper
	{
		public static void Run(Bitmap bitmap, string name)
		{
			if (!Path.HasExtension(name))
				name = $"{name}.png";

			bitmap.Save(name);
			Run(name);
		}

		public static void Run(string path, string arguments = null)
		{
			if (!Path.IsPathRooted(path) && Path.HasExtension(path))
				path = Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, path));

			var info = new ProcessStartInfo(path)
			{
				UseShellExecute = true
			};
			if (arguments != null)
			{
				info.Arguments = arguments;
			}

			Process.Start(info);
		}

		public static IEnumerable<string> PrintFormatted(this byte[] data, int width, int height, int channels = 1)
		{
			var sb = new StringBuilder();
			for (var y = 0; y < height; y++)
			{
				var line = data.AsSpan(y * width * channels, width * channels);
				for (var x = 0; x < width; x++)
				{
					var pixel = line.Slice(x * channels, channels);
					for (var i = 0; i < channels; i++)
					{
						sb.Append($"{pixel[i],4} ");
					}

					sb.Append("| ");
				}

				yield return sb.ToString();

				sb.Clear();
			}
		}

		public static IEnumerable<string> PrintFormatted(this float[] data, int width, int height, int channels = 1, bool rounded = true)
		{
			var sb = new StringBuilder();
			for (var y = 0; y < height; y++)
			{
				var line = data.AsSpan(y * width * channels, width * channels);
				for (var x = 0; x < width; x++)
				{
					var pixel = line.Slice(x * channels, channels);
					for (var i = 0; i < channels; i++)
					{
						var v = pixel[i];
						if (rounded)
							v = (float)Math.Round(v);
						sb.Append($"{v,4} ");
					}

					sb.Append("| ");
				}

				yield return sb.ToString();

				sb.Clear();
			}
		}

		public static IEnumerable<string> PrintFormatted(this int[] data, int width, int height, int channels = 1)
		{
			var min = int.MaxValue;
			var max = int.MinValue;
			for (var i = 0; i < data.Length; i++)
			{
				var v = data[i];
				min = Math.Min(min, v);
				max = Math.Max(max, v);
			}

			var len0 = min.ToString().Length;
			var len1 = max.ToString().Length;
			var len = Math.Max(len0, len1) + 1;

			var sb = new StringBuilder();
			for (var y = 0; y < height; y++)
			{
				var line = data.AsSpan(y * width * channels, width * channels);
				for (var x = 0; x < width; x++)
				{
					var pixel = line.Slice(x * channels, channels);
					sb.Append("|");
					for (var i = 0; i < channels; i++)
					{
						var v = $"{pixel[i]}".PadLeft(len);
						sb.Append($"{v} ");
					}
				}

				yield return sb.ToString();

				sb.Clear();
			}
		}

		public static void SaveFormatted(this byte[] data, string path, int width, int height, int channels = 1)
		{
			using var stream = new StreamWriter(path, false, Encoding.UTF8);
			foreach (var line in PrintFormatted(data, width, height, channels))
			{
				stream.WriteLine(line);
			}
		}

		public static void SaveFormatted(this float[] data, string path, int width, int height, int channels = 1, bool rounded = true)
		{
			using var stream = new StreamWriter(path, false, Encoding.UTF8);
			foreach (var line in PrintFormatted(data, width, height, channels, rounded))
			{
				stream.WriteLine(line);
			}
		}

		public static void SaveFormatted(this int[] data, string path, int width, int height, int channels = 1)
		{
			using var stream = new StreamWriter(path, false, Encoding.UTF8);
			foreach (var line in PrintFormatted(data, width, height, channels))
			{
				stream.WriteLine(line);
			}
		}

		public static void RunAsText(this byte[] data, int width, int height, int channels, string path)
		{
			if (!Path.IsPathRooted(path) && Path.HasExtension(path))
				path = Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, path));

			SaveFormatted(data, path, width, height, channels);
			Run("code", path);
		}

		public static void RunAsText(this float[] data, int width, int height, int channels, string path, bool rounded = true)
		{
			if (!Path.IsPathRooted(path) && Path.HasExtension(path))
				path = Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, path));

			SaveFormatted(data, path, width, height, channels, rounded);
			Run("code", path);
		}

		public static void RunAsText(this int[] data, int width, int height, int channels, string path)
		{
			if (!Path.IsPathRooted(path) && Path.HasExtension(path))
				path = Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, path));

			SaveFormatted(data, path, width, height, channels);
			Run("code", path);
		}

		public static void RunAs(this Bitmap image, string filename)
		{
			var path =
				Path.GetFullPath(
					Path.Combine(
						Environment.CurrentDirectory,
						filename));
			image.Save(path);
			Run(path);
		}

		public static void RunAs(this byte[] image, int width, int height, int channels, string filename)
		{
			using var bitmap = image.ToBitmap(width, height, channels);
			RunAs(bitmap, filename);
		}

		public static void RunAs(this float[] image, int width, int height, int channels, string filename)
		{
			var normalized = new byte[image.Length];
			image.Normalize(normalized, channels);
			using var bitmap = normalized.ToBitmap(width, height, channels);
			RunAs(bitmap, filename);
		}

		public static void RunAs(this int[] image, int width, int height, int channels, string filename)
		{
			var normalized = new byte[image.Length];
			image.Normalize(normalized, channels);
			using var bitmap = normalized.ToBitmap(width, height, channels);
			RunAs(bitmap, filename);
		}

		public static byte[] ToBytes(this Bitmap bitmap)
		{
			var w = bitmap.Width;
			var h = bitmap.Height;
			var bounds = new System.Drawing.Rectangle(0, 0, w, h);
			var data = bitmap.LockBits(bounds, ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);

			var result = new byte[h * w * 4];
			var resultHandler = GCHandle.Alloc(result, GCHandleType.Pinned);
			var resultPointer = resultHandler.AddrOfPinnedObject();

			var sourcePointer = data.Scan0;
			for (var y = 0; y < h; y++)
			{
				SharpDX.Utilities.CopyMemory(resultPointer, sourcePointer, w * 4);

				sourcePointer = IntPtr.Add(sourcePointer, data.Stride);
				resultPointer = IntPtr.Add(resultPointer, w * 4);
			}

			bitmap.UnlockBits(data);

			resultHandler.Free();

			return result;
		}

		public static Bitmap ToBitmap(this byte[] data, int width, int height, int channels = 1)
		{
			var bitmap = new Bitmap(width, height);
			var bounds = new System.Drawing.Rectangle(0, 0, width, height);

			var bitmapData = bitmap.LockBits(bounds, ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb);
			var bitmapPointer = bitmapData.Scan0;

			// Grayscale
			if (channels == 1)
			{
				var p = 0;
				var source = new byte[height * width * 4];
				for (var i = 0; i < height * width; i++)
				{
					var value = data[i];
					source[p++] = value;
					source[p++] = value;
					source[p++] = value;
					source[p++] = 255;
				}

				data = source;
			}

			var dataHandle = GCHandle.Alloc(data, GCHandleType.Pinned);
			var dataPointer = dataHandle.AddrOfPinnedObject();

			for (var y = 0; y < height; y++)
			{
				SharpDX.Utilities.CopyMemory(bitmapPointer, dataPointer, width * 4);

				dataPointer = IntPtr.Add(dataPointer, width * 4);
				bitmapPointer = IntPtr.Add(bitmapPointer, width * 4);
			}

			bitmap.UnlockBits(bitmapData);

			dataHandle.Free();

			return bitmap;
		}

		public static void Dump(this int[] data, int width, int height)
		{
			Console.WriteLine("{");
			for (var y = 0; y < height; y++)
			{
				for (var x = 0; x < width; x++)
				{
					var d = y * width + x;
					Console.Write($"{data[d]}, ");
				}
				Console.WriteLine();
			}
			Console.WriteLine("}");
		}
	}
}
