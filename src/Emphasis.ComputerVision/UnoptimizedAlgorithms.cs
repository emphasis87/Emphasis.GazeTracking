﻿using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Threading;
using Emphasis.ComputerVision.Primitives;
using RBush;

namespace Emphasis.ComputerVision
{
	public static partial class UnoptimizedAlgorithms
	{
		private static readonly float[] GrayscaleMask = 
		{ 
			0.2126f, // R
			0.7152f, // G
			0.0722f, // B
			0
		};

		public static void Grayscale(int width, int height, byte[] source, byte[] grayscale)
		{
			for (var y = 0; y < height; y++)
			{
				for (var x = 0; x < width; x++)
				{
					var g =
						source[y * (width * 4) + x * 4 + 0] * GrayscaleMask[0] +
						source[y * (width * 4) + x * 4 + 1] * GrayscaleMask[1] +
						source[y * (width * 4) + x * 4 + 2] * GrayscaleMask[2] +
						source[y * (width * 4) + x * 4 + 3] * GrayscaleMask[3];

					var d = y * width + x;
					grayscale[d] = Convert.ToByte(Math.Min(g, 255));
				}
			}
		}

		public static void GrayscaleEq(int width, int height, byte[] source, byte[] grayscale)
		{
			for (var y = 0; y < height; y++)
			{
				for (var x = 0; x < width; x++)
				{
					var d = y * width + x;
					var gs =
						source[y * (width * 4) + x * 4 + 0] +
						source[y * (width * 4) + x * 4 + 1] +
						source[y * (width * 4) + x * 4 + 2];

					var g = Math.Min(255, gs / 3);
					grayscale[d] = (byte)g;
				}
			}
		}

		public static void Sort(Span<int> source, Span<int> other, Span<int> sourceBuffer, Span<int> otherBuffer)
		{
			var n = source.Length;
			switch (n)
			{
				case 0:
				case 1:
					return;
				case 2:
				{
					ref var a = ref source[0];
					ref var b = ref source[1];
					if (a > b)
					{
						var c = a;
						a = b;
						b = c;
						var d = other[1];
						other[1] = other[0];
						other[0] = d;
					}
					return;
				}
			}

			var n2 = n >> 1;
			var s0 = source.Slice(0, n2);
			var s1 = source.Slice(n2, n - n2);
			var o0 = other.Slice(0, n2);
			var o1 = other.Slice(n2, n - n2);
			var sb0 = sourceBuffer.Slice(0, n2);
			var sb1 = sourceBuffer.Slice(n2, n - n2);
			var ob0 = otherBuffer.Slice(0, n2);
			var ob1 = otherBuffer.Slice(n2, n - n2);
			
			Sort(s0, o0, sb0, ob0);
			Sort(s1, o1, sb1, ob1);
			
			s0.CopyTo(sb0);
			s1.CopyTo(sb1);
			o0.CopyTo(ob0);
			o1.CopyTo(ob1);

			var p0 = 0;
			var p1 = 0;
			for (var i = 0; i < n; i++)
			{
				if (p1 >= n - n2 || (p0 < n2 && sb0[p0] <= sb1[p1]))
				{
					source[i] = sb0[p0];
					other[i] = ob0[p0];
					p0++;
				}
				else
				{
					source[i] = sb1[p1];
					other[i] = ob1[p1];
					p1++;
				}
			}
		}

		public static void LinePrefixSum(int width, int height, byte[] source, int sourceChannels, int[] linePrefixSums)
		{
			Span<int> prefix = stackalloc int[sourceChannels];
			for (var y = 0; y < height; y++)
			{
				for (var x = 0; x < width; x++)
				{
					for (var c = 0; c < sourceChannels; c++)
					{
						var d = y * width * sourceChannels + x * sourceChannels + c;
						var v = source[d];
						var r = prefix[c] += v;
						linePrefixSums[d] = r;
					}
				}
			}
		}

		public static void BoxBlur(int width, int height, int[] linePrefixSums, int sourceChannels, byte[] box, int windowSize)
		{
			for (var y = 0; y < height; y++)
			{
				for (var x = 0; x < width; x++)
				{
					var x1 = Math.Max(0, x - windowSize);
					var x2 = Math.Min(width - 1, x + windowSize);
					for (var c = 0; c < sourceChannels; c++)
					{
						var d = y * width * sourceChannels + x * sourceChannels + c;
						var y1 = Math.Max(0, y - windowSize);
						var y2 = Math.Min(height - 1, y + windowSize);
						var sum = 0;
						for (var yi = y1; yi <= y2; yi++)
						{
							var d1 = yi * width * sourceChannels + x1 * sourceChannels + c;
							var d2 = yi * width * sourceChannels + x2 * sourceChannels + c;
							var diff = linePrefixSums[d2] - linePrefixSums[d1];
							sum += diff;
						}

						var avg = sum / ((y2 - y1) * (x2 - x1));
						box[d] = (byte)Math.Min(255, avg);
					}
				}
			}
		}

		public static void Background(int width, int height, byte[] source, int sourceChannels, byte[] grayscale, byte[] background, int windowSize)
		{
			var ws = windowSize;
			var wa = ws >> 1;
			var wb = ws - wa;
			var ws2 = (ws * ws) >> 1;
			
			Span<int> window = stackalloc int[ws * ws];
			Span<int> indexes = stackalloc int[ws * ws];
			Span<int> coordinates = stackalloc int[ws * ws * 2];
			Span<int> windowBuffer = stackalloc int[ws * ws];
			Span<int> indexesBuffer = stackalloc int[ws * ws];
			Span<int> hist = stackalloc int[ws * ws];

			for (var y = 0; y < height; y++)
			{
				for (var x = 0; x < width; x++)
				{
					var i = 0;
					for (var yi = 0; yi < ws; yi++)
					{
						var yn = y + (yi - wa);
						if (yn < 0 || yn >= height)
							continue;

						for (var xi = 0; xi < ws; xi++)
						{
							var xn = x + (xi - wa);
							if (xn < 0 || xn >= width)
								continue;

							var index = yn * width + xn;
							window[i] = grayscale[index];
							indexes[i] = i;
							coordinates[2 * i] = yn;
							coordinates[2 * i + 1] = xn;
							i++;
						}
					}

					Sort(window.Slice(0, i), indexes.Slice(0, i), windowBuffer, indexesBuffer);

					var j0 = 0;
					var j1 = 0;

					hist.Fill(0);
					var hmax = 0;
					var hi = 0;

					for (var k = 0; k < i; k++)
					{
						if (j0 < k)
							j0 = k;

						var g0 = window[k];
						j1 = Math.Max(j0, j1);
						for (var j = j1; j < i; j++)
						{
							if (Math.Abs(window[j] - g0) < 50)
							{
								j1 = j;
								continue;
							}

							break;
						}

						var i1 = indexes[k];
						var y1 = coordinates[2 * i1];
						var x1 = coordinates[2 * i1 + 1];
						var d1 = y1 * width * sourceChannels + x1 * sourceChannels;
						if (k > 0)
						{
							var i2 = indexes[k - 1];
							var y2 = coordinates[2 * i2];
							var x2 = coordinates[2 * i2 + 1];
							var d2 = y2 * width * sourceChannels + x2 * sourceChannels;
							if (IsSameColor(source, sourceChannels, d1, d2, 0))
							{
								hist[k] = hist[k - 1];
								continue;
							}
						}

						var h = 0;
						for (var j = j0; j <= j1; j++)
						{
							if (j == k)
							{
								h++;
								continue;
							}

							var i2 = indexes[j];
							var y2 = coordinates[2 * i2];
							var x2 = coordinates[2 * i2 + 1];
							var d2 = y2 * width * sourceChannels + x2 * sourceChannels;
							if (!IsSameColor(source, sourceChannels, d1, d2, 30))
								continue;
							
							h++;
						}

						if (h > hmax)
						{
							hmax = h;
							hi = k;

							if (h > ws2)
								break;
						}
					}

					var dc = y * width * sourceChannels + x * sourceChannels;

					var i3 = indexes[hi];
					var y3 = coordinates[2 * i3];
					var x3 = coordinates[2 * i3 + 1];
					var d3 = y3 * width * sourceChannels + x3 * sourceChannels;
					for (var c = 0; c < sourceChannels; c++)
					{
						background[dc + c] = source[d3 + c];
					}
				}
			}
		}

		public static bool IsSameColor(byte[] source, int channels, int i1, int i2, int channelTolerance)
		{
			var isSameColor = true;
			for (var c = 0; c < channels; c++)
			{
				if (Math.Abs(source[i1 + c] - source[i2 + c]) > channelTolerance)
				{
					isSameColor = false;
					break;
				}
			}

			return isSameColor;
		}

		public static bool IsSameColor(Span<byte> pixel, byte[] other, int channels, int i1, int channelTolerance)
		{
			var isSameColor = true;
			for (var c = 0; c < channels; c++)
			{
				if (Math.Abs(pixel[c] - other[i1 + c]) > channelTolerance)
				{
					isSameColor = false;
					break;
				}
			}

			return isSameColor;
		}

		public static bool IsSameColor(Span<byte> p0, Span<byte> p1, int channels, int channelTolerance)
		{
			var isSameColor = true;
			for (var c = 0; c < channels; c++)
			{
				if (Math.Abs(p0[c] - p1[c]) > channelTolerance)
				{
					isSameColor = false;
					break;
				}
			}

			return isSameColor;
		}

		public static bool IsSameColor(Span<byte> p0, Span<byte> p1, int channels, int channelTolerance, out int colorDifference)
		{
			var isSameColor = true;
			colorDifference = 0;
			for (var c = 0; c < channels; c++)
			{
				var diff = Math.Abs(p0[c] - p1[c]);
				if (diff > channelTolerance)
				{
					isSameColor = false;
					break;
				}
				colorDifference += diff;
			}

			return isSameColor;
		}

		public static bool IsSameColor(Span<byte> pixel, int[] other, int channels, int i1, int channelTolerance)
		{
			var isSameColor = true;
			for (var c = 0; c < channels; c++)
			{
				if (Math.Abs(pixel[c] - other[i1 + c]) > channelTolerance)
				{
					isSameColor = false;
					break;
				}
			}

			return isSameColor;
		}

		public static void Enlarge2Interpolated(int width, int height, byte[] source, byte[] destination, int channels = 4)
		{
			for (var y = 0; y < height - 1; y++)
			{
				for (var x = 0; x < width - 1; x++)
				{
					for (var c = 0; c < channels; c++)
					{
						var d1 = (2 * y + 0) * 2 * width * channels + (2 * x + 0) * channels + c;
						var d2 = (2 * y + 0) * 2 * width * channels + (2 * x + 1) * channels + c;
						var d3 = (2 * y + 1) * 2 * width * channels + (2 * x + 0) * channels + c;
						var d4 = (2 * y + 1) * 2 * width * channels + (2 * x + 1) * channels + c;

						var s1 = (y + 0) * width * channels + (x + 0) * channels + c;
						var s2 = (y + 0) * width * channels + (x + 1) * channels + c;
						var s3 = (y + 1) * width * channels + (x + 0) * channels + c;
						var s4 = (y + 1) * width * channels + (x + 1) * channels + c;

						destination[d1] = source[s1];
						destination[d2] = (byte) ((source[s1] + source[s2]) >> 1);
						destination[d3] = (byte) ((source[s1] + source[s3]) >> 1);
						destination[d4] = (byte) ((source[s1] + source[s4]) >> 1);
					}
				}
			}
		}

		public static void Enlarge2(int width, int height, byte[] source, byte[] destination, int channels = 4)
		{
			for (var y = 0; y < height - 1; y++)
			{
				for (var x = 0; x < width - 1; x++)
				{
					for (var c = 0; c < channels; c++)
					{
						var d1 = (2 * y + 0) * 2 * width * channels + (2 * x + 0) * channels + c;
						var d2 = (2 * y + 0) * 2 * width * channels + (2 * x + 1) * channels + c;
						var d3 = (2 * y + 1) * 2 * width * channels + (2 * x + 0) * channels + c;
						var d4 = (2 * y + 1) * 2 * width * channels + (2 * x + 1) * channels + c;

						var s1 = (y + 0) * width * channels + (x + 0) * channels + c;
						var s2 = (y + 0) * width * channels + (x + 1) * channels + c;
						var s3 = (y + 1) * width * channels + (x + 0) * channels + c;
						var s4 = (y + 1) * width * channels + (x + 1) * channels + c;

						destination[d1] = source[s1];
						destination[d2] = source[s1];
						destination[d3] = source[s1];
						destination[d4] = source[s1];
					}
				}
			}
		}

		//constant float gauss[3][3] = 
		//{   
		//	{ 0.0625, 0.1250, 0.0625 },
		//	{ 0.1250, 0.2500, 0.1250 },
		//	{ 0.0625, 0.1250, 0.0625 },
		//};

		public static void Gauss(int width, int height, byte[] source, byte[] destination, int channels = 4)
		{
			for (var y = 0; y < height; y++)
			{
				var i = Clamp(y, height);
				for (var x = 0; x < width; x++)
				{
					var j = Clamp(x, width);
					for (var c = 0; c < channels; c++)
					{
						var result =
							source[(i - 1) * (width * channels) + (j - 1) * channels + c] * 0.0625f +
							source[(i - 1) * (width * channels) + (j + 0) * channels + c] * 0.1250f +
							source[(i - 1) * (width * channels) + (j + 1) * channels + c] * 0.0625f +
							source[(i + 0) * (width * channels) + (j - 1) * channels + c] * 0.1250f +
							source[(i + 0) * (width * channels) + (j + 0) * channels + c] * 0.2500f +
							source[(i + 0) * (width * channels) + (j + 1) * channels + c] * 0.1250f +
							source[(i + 1) * (width * channels) + (j - 1) * channels + c] * 0.0625f +
							source[(i + 1) * (width * channels) + (j + 0) * channels + c] * 0.1250f +
			 				source[(i + 1) * (width * channels) + (j + 1) * channels + c] * 0.0625f;

						var d = y * (width * channels) + x * channels + c;
						destination[d] = Convert.ToByte(Math.Min(255, result));
					}
				}
			}
		}

		private static readonly float[,] SobelDxMask3 = new float[,]
		{
			{ +1, +0, -1 },
			{ +2, +0, -2 },
			{ +1, +0, -1 },
		};

		private static readonly float[,] SobelDyMask3 = new float[,]
		{
			{ +1, +2, +1 },
			{  0,  0,  0 },
			{ -1, -2, -1 },
		};

		private static readonly float[,] SobelDxMask5 = new float[,]
		{
			{ +1, +2, +0, -2, -1 },
			{ +4, +8, +0, -8, -4 },
			{ +6,+12, +0,-12, -6 },
			{ +4, +8, +0, -8, -4 },
			{ +1, +2, +0, -2, -1 },
		};

		private static readonly float[,] SobelDyMask5 = new float[,]
		{
			{ +1, +4, +6, +2, +1 },
			{ +2, +8,+12, +8, +4 },
			{ +0, +0, +0, +0, +0 },
			{ -2, -8,-12, -8, -4 },
			{ -1, -4, -6, -4, -1 },
		};

		//private static readonly float[,] SobelDxMask3 = new float[,]
		//{
		//	{  +3, +0,  -3 },
		//	{ +10, +0, -10 },
		//	{  +3, +0,  -3 },
		//};

		//private static readonly float[,] SobelDyMask3 = new float[,]
		//{
		//	{ +3, +10, +3 },
		//	{  0,  0,  0 },
		//	{ -3, -10, -10 },
		//};

		public static void Sobel3(int width, int height, byte[] source, float[] dx, float[] dy, float[] gradient, float[] angle, byte[] neighbors, int channels = 4)
		{
			for (var y = 0; y < height; y++)
			{
				var i = Clamp(y, height);
				for (var x = 0; x < width; x++)
				{
					var j = Clamp(x, width);

					var idx = 0.0f;
					var idxa = 0.0f;
					var idy = 0.0f;
					var idya = 0.0f;
					var cdx = 0.0f;
					var cdy = 0.0f;

					for (var c = 0; c < channels; c++)
					{
						for (var iy = 0; iy < 3; iy++)
						{
							for (var ix = 0; ix < 3; ix++)
							{
								cdx += source[(i + (iy - 1)) * (width * channels) + (j + (ix - 1)) * channels + c] * SobelDxMask3[iy, ix];
								cdy += source[(i + (iy - 1)) * (width * channels) + (j + (ix - 1)) * channels + c] * SobelDyMask3[iy, ix];
							}
						}

						cdx /= 8;
						cdy /= 8;

						var cdxAbs = MathF.Abs(cdx);
						if (cdxAbs > idxa)
						{
							idx = cdx;
							idxa = cdxAbs;
						}

						var cdyAbs = MathF.Abs(cdy);
						if (cdyAbs > idya)
						{
							idy = cdy;
							idya = cdyAbs;
						}
					}

					var d = y * width + x;

					dx[d] = idx;
					dy[d] = idy;

					// Up to sqrt(255*4 * 255*4 + 255*4 * 255*4) = 1442
					var g = Gradient(idx, idy);
					gradient[d] = g;

					var a = GradientAngle(idx, idy);
					angle[d] = a;

					var (direction, count, w0, w1, w2) = GradientNeighbors(a);
					var dn = y * width * 5 + x * 5;
					neighbors[dn] = direction;
					neighbors[dn + 1] = count;
					neighbors[dn + 2] = w0;
					neighbors[dn + 3] = w1;
					neighbors[dn + 4] = w2;
				}
			}
		}

		public static void Sobel5(int width, int height, byte[] source, float[] dx, float[] dy, float[] gradient, float[] angle, byte[] neighbors, int channels = 4)
		{
			for (var y = 0; y < height; y++)
			{
				for (var x = 0; x < width; x++)
				{
					var idx = 0.0f;
					var idxa = 0.0f;
					var idy = 0.0f;
					var idya = 0.0f;
					var cdx = 0.0f;
					var cdy = 0.0f;

					for (var c = 0; c < channels; c++)
					{
						for (var iy = 0; iy < 5; iy++)
						{
							for (var ix = 0; ix < 5; ix++)
							{
								var yn = ClampToEdge(y + (iy - 2), height);
								var xn = ClampToEdge(x + (ix - 2), width);
								cdx += source[yn * width * channels + xn * channels + c] * SobelDxMask5[iy, ix];
								cdy += source[yn * width * channels + xn * channels + c] * SobelDyMask5[iy, ix];
							}
						}

						cdx /= 16;
						cdy /= 16;

						var cdxAbs = MathF.Abs(cdx);
						if (cdxAbs > idxa)
						{
							idx = cdx;
							idxa = cdxAbs;
						}

						var cdyAbs = MathF.Abs(cdy);
						if (cdyAbs > idya)
						{
							idy = cdy;
							idya = cdyAbs;
						}
					}

					var d = y * width + x;

					dx[d] = idx;
					dy[d] = idy;

					// Up to sqrt(255*4 * 255*4 + 255*4 * 255*4) = 1442
					var g = Gradient(idx, idy);
					gradient[d] = g;

					var a = GradientAngle(idx, idy);
					angle[d] = a;

					var (direction, count, w0, w1, w2) = GradientNeighbors(a);
					var dn = y * width * 5 + x * 5;
					neighbors[dn] = direction;
					neighbors[dn + 1] = count;
					neighbors[dn + 2] = w0;
					neighbors[dn + 3] = w1;
					neighbors[dn + 4] = w2;
				}
			}
		}

		public static float GradientAngle(float dx, float dy)
		{
			var a = MathF.Atan2(dy, -dx) / MathF.PI + 2.0f;
			if (a >= 2)
				a -= 2;
			a *= 180;
			return a;
		}

		public static float Gradient(float dx, float dy)
		{
			return MathF.Sqrt(dx * dx + dy * dy);
		}

		public static (byte direction, byte count, byte weight0, byte weight1, byte weight2) GradientNeighbors(float angle)
		{
			// Shift the angle by half of a step
			var a = angle + 11.25f; // 360/32
			if (a >= 360)
				a -= 360;

			var b = Convert.ToByte(MathF.Ceiling(a / 22.5f));
			var c = b % 2;
			var count = (byte)(c + 2);

			var direction = Convert.ToByte(MathF.Ceiling(a / 45.0f));
			if (direction >= 8)
				direction -= 8;

			if (c == 1)
				return (direction, count, 25, 50, 25);

			return (direction, count, 50, 50, 0);
		}

		private static readonly int[,] Neighbors =
		{
			// x   y
			{ +1,  0 }, //   0° E
			{ +1, -1 }, //  45° NE
			{  0, -1 }, //  90° N
			{ -1, -1 }, // 135° NW
			{ -1,  0 }, // 180° W
			{ -1, +1 }, // 225° SW
			{  0, +1 }, // 270° S
			{  1, +1 }, // 315° SE
		};

		public static void NonMaximumSuppression(
			int width, 
			int height, 
			float[] gradient,
			float[] angle,
			byte[] neighbors, 
			float[] destination, 
			float[] cmp1, 
			float[] cmp2)
		{
			for (var y = 0; y < height; y++)
			{
				for (var x = 0; x < width; x++)
				{
					if (x == 0 || x == width - 1 || y == 0 || y == height - 1)
						continue;

					var d = y * width + x;
					destination[d] = 0;

					var g = gradient[d];
					if (g < 15)
						continue;

					var dn = y * width * 5 + x * 5;
					var neighbor = neighbors[dn];
					var count = neighbors[dn + 1];

					var g1 = 0f;
					var g2 = 0f;

					for (var i = 0; i < count; i++)
					{
						var weight = neighbors[dn + 2 + i];
						var n = neighbor - i;
						if (n < 0)
							n += 8;

						var dx1 = Neighbors[n, 0];
						var dy1 = Neighbors[n, 1];
						var dx2 = -dx1;
						var dy2 = -dy1;

						var d1 = (y + dy1) * width + (x + dx1);
						var d2 = (y + dy2) * width + (x + dx2);

						g1 += weight * gradient[d1];
						g2 += weight * gradient[d2];
					}

					g1 *= 0.01f;
					g2 *= 0.01f;

					cmp1[d] = g1;
					cmp2[d] = g2;

					if (g >= g1 && g >= g2)
						destination[d] = g;
				}
			}
		}

		public static void StrokeWidthTransform(
			int width,
			int height,
			byte[] source,
			float[] gradient,
			float[] edges,
			float[] angles,
			float[] dx,
			float[] dy,
			int[] swt0,
			int[] swt1,
			int sourceChannels = 4,
			int rayLength = 20,
			int colorDifference = 50,
			bool useStrokeColor = false)
		{
			// Prefix scan edges
			var edgeList = new List<int>();
			for (var y = 0; y < height; y++)
			{
				for (var x = 0; x < width; x++)
				{
					var d = y * width + x;
					var g = edges[d];
					if (g > 0)
					{
						edgeList.Add(x);
						edgeList.Add(y);
					}
				}
			}

			// Find the stroke width in positive direction
			var swtList0 = new List<int>();
			StrokeWidthTransform(width, height, source, gradient,edges, angles, dx, dy, swt0, rayLength, true, edgeList, swtList0, 
				channels: sourceChannels,
				colorDifference: colorDifference,
				useStrokeColor: useStrokeColor);

			// Find the stroke width in positive direction
			var swtList1 = new List<int>();
			StrokeWidthTransform(width, height, source, gradient, edges, angles, dx, dy, swt1, rayLength, false, edgeList, swtList1, 
				channels: sourceChannels, 
				colorDifference: colorDifference,
				useStrokeColor: useStrokeColor);
		}

		public static float Hypot(int width, int height)
		{
			return MathF.Sqrt(width * width + height * height);
		}

		public static void StrokeWidthTransform(
			int width,
			int height,
			byte[] source,
			float[] gradient,
			float[] edges,
			float[] angles,
			float[] dx,
			float[] dy,
			int[] swt,
			int rayLength,
			bool direction,
			List<int> edgeList,
			List<int> swtList,
			int channels = 4,
			int colorDifference = 50,
			bool useStrokeColor = false)
		{
			var dir = direction ? 1 : -1;

			int x, y, d, len, sw, cx, cy, mx, my, ix, iy;
			float a, idx, idy, idxa, idya, ex, ey;

			void InitializeLine(float rdx = 1.0f, float rdy = 1.0f)
			{
				d = y * width + x;

				// Differential on x and y-axis
				idx = dx[d] * rdx;
				idy = dy[d] * rdy;

				// Sign of the differential in the direction (black on white or white on black)
				ix = dir * MathF.Sign(idx);
				iy = dir * MathF.Sign(idy);

				// The size of the differential
				idxa = MathF.Abs(idx);
				idya = MathF.Abs(idy);

				ResetLine();
			}

			void ResetLine()
			{
				// The current error
				ex = idxa;
				ey = idya;

				// The current position
				cx = x;
				cy = y;
			}

			void Advance()
			{
				// Move by 1 (direct neighbor is likely an edge in the same direction)
				if (ex >= ey)
				{
					ey += idya;
					cx += ix;
				}
				else
				{
					ex += idxa;
					cy += iy;
				}
			}

			void AddEdge(int cd, float ra, int ci)
			{
				// Check that the found edge is roughly opposite
				var ca = angles[cd] + ra;
				if (ca > 360)
					ca -= 360;
				var cad = MathF.Abs(180 - MathF.Abs(a - ca));
				if (cad < 45)
				{
					swtList.Add(x);
					swtList.Add(y);
					swtList.Add(ci);
					var sw = (int)Hypot(Math.Abs(x - cx), Math.Abs(y - cy));
					swtList.Add(sw);
				}
			}

			Span<byte> src = stackalloc byte[channels];

			var rp = new float[,]
			{
				{0.75f, 1.00f, 15},
				{1.00f, 1.00f, 0},
				{1.00f, 0.75f, 345},
			};
			var rc = rp.Length / 3;

			var swtItemSize = 4;

			var en = edgeList.Count;
			for (var i = 0; i < en; i += 2)
			{
				for (var r = 0; r < rc; r++)
				{
					var rdx = rp[r, 0];
					var rdy = rp[r, 1];
					var ra = rp[r, 2];

					x = edgeList[i];
					y = edgeList[i + 1];

					InitializeLine(rdx, rdy);

					a = angles[d];

					// The indexing limits
					mx = ix > 0 ? width : -1;
					my = iy > 0 ? height : -1;

					Advance();

					if (cx == mx || cy == my)
						continue;

					if (useStrokeColor)
					{
						for (var c = 0; c < channels; c++)
						{
							src[c] = source[cy * width * channels + cx * channels + c];
						}
					}

					// Find and collect opposing edges
					for (var ci = 2; ci < rayLength; ci++)
					{
						Advance();

						if (cx == mx || cy == my)
							break;

						// The current distance
						var cd = cy * width + cx;
						var cg = edges[cd];

						if (cg > 0)
						{
							AddEdge(cd, ra, ci);
							break;
						}

						if (useStrokeColor)
						{
							var isSameColor = true;
							for (var c = 0; c < channels; c++)
							{
								var dst = source[cy * width * channels + cx * channels + c];
								var dif = Math.Abs(dst - src[c]);
								if (dif > colorDifference)
								{
									isSameColor = false;
									break;
								}
							}

							if (!isSameColor)
							{
								AddEdge(cd, ra, ci);
								break;
							}
						}
					}
				}
			}

			// Fill in the strokes
			var sn = swtList.Count;
			for (var i = 0; i < sn; i += swtItemSize)
			{
				x = swtList[i];
				y = swtList[i + 1];
				len = swtList[i + 2];
				sw = swtList[i + 3];

				InitializeLine();

				Advance();

				for (var ci = 1; ci < len; ci++)
				{
					// The current distance
					var cd = cy * width + cx;
					var cs = swt[cd];
					// Set the stroke width to the lowest found
					if (cs > len)
						swt[cd] = sw;

					Advance();
				}
			}

			for (int i = 0, j = 0; i < sn; j++, i += swtItemSize)
			{
				x = swtList[i];
				y = swtList[i + 1];
				len = swtList[i + 2];
				
				InitializeLine();

				Advance();

				// Find the median stroke width for the ray
				var sm = new List<int>();
				for (var ci = 1; ci < len; ci++)
				{
					// The current distance
					var cd = cy * width + cx;
					if (cy < 0 || cy >= height || cx < 0 || cx >= width)
						throw new ArgumentOutOfRangeException();
					var cs = swt[cd];
					sm.Add(cs);

					Advance();
				}

				var median = Median(sm);

				ResetLine();

				Advance();

				// Cap the stroke width to the ray's median
				for (var ci = 1; ci < len; ci++)
				{
					// The current distance
					var cd = cy * width + cx;
					if (cy < 0 || cy >= height || cx < 0 || cx >= width)
						throw new ArgumentOutOfRangeException();
					var cs = swt[cd];
					if (cs > median)
						swt[cd] = median;

					Advance();
				}
			}
		}

		public static void PrepareComponents(int[] swt, int[] components)
		{
			var n = components.Length;
			for (var i = 0; i < swt.Length; i++)
			{
				var stroke = swt[i];
				components[i] = stroke < int.MaxValue ? i : n + i;
			}
		}

		public static void IndexComponents(int[] components)
		{
			var n = components.Length;
			for (var i = 0; i < n; i++)
			{
				components[i] = i;
			}
		}

		public static int ModOverflow(int i, int mod)
		{
			if (i >= mod)
				i -= mod;
			return i;
		}

		public static int ColorComponentsWatershed(
			int width,
			int height,
			int[] swt,
			int[] coloring)
		{
			var rounds = 0;
			var isComplete = false;
			while (!isComplete)
			{
				rounds++;
				isComplete = true;
				for (var y = 0; y < height; y++)
				{
					for (var x = 0; x < width; x++)
					{
						var d = y * width + x;
						var cn = ColorComponentByStrokeWidth(
							width, height, swt, coloring,
							x, y, d);

						if (cn != coloring[d])
						{
							coloring[d] = cn;
							isComplete = false;
						}
					}
				}
			}

			return rounds;
		}

		public static int ColorComponentsFixedPoint(
			int width,
			int height,
			int[] swt,
			int[] coloring)
		{
			var n = height * width;
			var rounds = 0;
			var isColored = false;
			while (!isColored)
			{
				rounds++;
				isColored = true;
				for (var y = 0; y < height; y++)
				{
					for (var x = 0; x < width; x++)
					{
						var d = y * width + x;
						var c = coloring[d];
						var cn = ColorComponentByStrokeWidth(
							width, height, swt, coloring,
							x, y, d);

						if (cn  < c)
						{
							for (var i = 0; i < 4; i++)
							{
								cn = coloring[ModOverflow(cn, n)];
							}

							coloring[d] = cn;
							isColored = false;
						}
					}
				}
			}

			return rounds;
		}

		public static int ColorComponentsFixedPointBackPropagation(
			int width,
			int height,
			int[] swt,
			int[] coloring)
		{
			var n = height * width;
			var rounds = 0;

			var isColored = false;
			while (!isColored)
			{
				rounds++;
				isColored = true;
				for (var y = 0; y < height; y++)
				{
					for (var x = 0; x < width; x++)
					{
						var d = y * width + x;
						var c0 = coloring[d];
						var cn = ColorComponentByStrokeWidth(
							width, height, swt, coloring, 
							x, y, d);

						if (cn >= c0)
							continue;

						for (var i = 0; i < 4; i++)
						{
							var cq = coloring[ModOverflow(cn, n)];
							cn = cq;
						}

						AtomicMin(ref coloring[ModOverflow(c0, n)], cn);
						AtomicMin(ref coloring[d], cn);

						coloring[d] = cn;
						isColored = false;
					}
				}
			}

			return rounds;
		}

		public static int ColorComponentByStrokeWidth(
			int width, 
			int height,
			int[] swt,
			int[] coloring,
			int x0, 
			int y0,
			int d)
		{
			var c = int.MaxValue;
			var c0 = coloring[d];
			var s0 = swt[d];

			for (var yi = -1; yi <= 1; yi++)
			{
				var y1 = y0 + yi;
				if (y1 < 0 || y1 >= height) 
					continue;

				for (var xi = -1; xi <= 1; xi++)
				{
					if (xi == 0 && yi == 0)
						continue;

					var x1 = x0 + xi;
					if (x1 < 0 || x1 >= width)
						continue;

					var d1 = y1 * width + x1;
					var c1 = coloring[d1];
					var s1 = swt[d1];

					if (s0 != int.MaxValue && s1 != int.MaxValue)
					{
						var smin = Math.Min(s0, s1);
						var smax = Math.Max(s0, s1);
						if (smax <= smin * 3)
							c = Math.Min(c, c1);
					}
				}
			}

			return Math.Min(c, c0);
		}

		public static int ColorComponentsFixedPointByColorSimilarity(
			int width,
			int height,
			int[] swt,
			int[] coloring,
			byte[] source,
			byte[] background,
			int channels,
			int[] componentIndexByColoring,
			Component[] components,
			int colorPerChannelTolerance = 30,
			bool isSourceEnlarged = false)
		{
			var n = height * width;
			var rounds = 0;
			var isColored = false;
			while (!isColored)
			{
				rounds++;
				isColored = true;
				for (var y = 0; y < height; y++)
				{
					for (var x = 0; x < width; x++)
					{
						var d = y * width + x;
						var c = coloring[d];
						var cn = ColorComponentByColorSimilarity(
							width, height, n, swt, coloring, source, background, channels, componentIndexByColoring, components,
							x, y, d,
							colorPerChannelTolerance: colorPerChannelTolerance,
							isSourceEnlarged: isSourceEnlarged);

						if (cn < c)
						{
							// Fixed point: f(f(f(f(x))))
							for (var i = 0; i < 4; i++)
							{
								cn = coloring[ModOverflow(cn, n)];
							}

							coloring[d] = cn;
							isColored = false;
						}
					}
				}
			}

			return rounds;
		}

		public static int ColorComponentByColorSimilarity(
			int width,
			int height,
			int n,
			int[] swt,
			int[] coloring,
			byte[] source,
			byte[] background,
			int channels,
			int[] componentIndexByColoring,
			Component[] components,
			int x0,
			int y0,
			int d,
			int colorPerChannelTolerance = 30,
			bool isSourceEnlarged = false)
		{
			var c = int.MaxValue;
			var c0 = coloring[d];
			var s0 = swt[d];

			Span<byte> p0 = stackalloc byte[channels];
			Span<byte> p1 = stackalloc byte[channels];
			p0.PixelAt(source, channels, width, x0, y0);

			Span<byte> b0 = stackalloc byte[channels];
			if (isSourceEnlarged)
				b0.PixelAt(background, channels, width >> 1,  x0 >> 1 , y0 >> 1);
			else
				b0.PixelAt(background, channels, width, x0, y0);

			var isBackground = IsSameColor(p0, b0, channels, colorPerChannelTolerance);

			var colorDifference = int.MaxValue;
			var cn = int.MaxValue;

			for (var yi = -1; yi <= 1; yi++)
			{
				var y1 = y0 + yi;
				if (y1 < 0 || y1 >= height)
					continue;

				for (var xi = -1; xi <= 1; xi++)
				{
					if (xi == 0 && yi == 0)
						continue;

					var x1 = x0 + xi;
					if (x1 < 0 || x1 >= width)
						continue;

					var d1 = y1 * width + x1;
					var c1 = coloring[d1];

					// The neighbor has the same color or is not connected
					if (c0 == c1 || c1 >= n)
						continue;

					var s1 = swt[d1];
					// Both pixels are strokes
					if (s0 != int.MaxValue && s1 != int.MaxValue)
					{
						var smin = Math.Min(s0, s1);
						var smax = Math.Max(s0, s1);
						// Too high stroke difference should not be connected
						if (smax <= smin * 3)
						{
							c = Math.Min(c, c1);
							continue;
						}
					}

					// Do not connect background pixels to the component
					if (isBackground)
						continue;

					p1.PixelAt(source, channels, width, x1, y1);
					var isSameColor = IsSameColor(p0, p1, channels, colorPerChannelTolerance, out var colorDifference1);
					if (!isSameColor || colorDifference < colorDifference1)
						continue;

					colorDifference = colorDifference1;
					c = Math.Min(c, c1);
					cn = c1;
				}
			}

			if (c0 < n && c < c0 && c == cn)
			{
				// Merge components
				var ci0 = componentIndexByColoring[c0];
				if (ci0 > 0)
					AtomicMin(ref components[ci0].ParentColoring, c);
			}

			return Math.Min(c, c0);
		}

		public static void PixelAt(this Span<byte> pixel, byte[] source, int channels, int width, int x, int y)
		{
			var d = y * width * channels + x * channels;
			for (var c = 0; c < channels; c++)
				pixel[c] = source[d + c];
		}

		public static int ComponentAnalysis(
			int width, 
			int height,
			byte[] source,
			int[] swt, 
			int[] coloring, 
			int[] componentIndexByColoring, 
			int[] componentItems,
			int[] componentSwtItems,
			Component[] components,
			int componentsLimit, 
			int componentSizeLimit,
			int sourceChannels = 4)
		{
			Array.Fill(componentIndexByColoring, -1);

			for (var ci = 0; ci < componentsLimit; ci++)
			{
				ref var c = ref components[ci];
				c.Initialize();
			}

			var n = coloring.Length;
			var count = -1;
			var i = 0;
			for (var y = 0; y < height; y++)
			{
				for (var x = 0; x < width; x++, i++)
				{
					var color = coloring[i];
					if (color >= n)
						continue;

					if (color == i)
						continue;

					var ci = componentIndexByColoring[color];
					if (ci == -1)
					{
						ci = Interlocked.Increment(ref count);
						if (ci >= componentsLimit)
							return count;

						componentIndexByColoring[color] = ci;
						ref var c0 = ref components[ci];
						c0.Coloring = color;
					}

					ref var c = ref components[ci];
					var size = Interlocked.Increment(ref c.Size);
					if (size >= componentSizeLimit)
						continue;

					var offset = ci * componentSizeLimit;
					var s = swt[i];
					if (s < int.MaxValue)
					{
						var swtSize = Interlocked.Increment(ref c.SwtSize);
						componentItems[offset + swtSize] = i;
						componentSwtItems[offset + swtSize] = s;

						Interlocked.Add(ref c.SwtSum, s);
					}

					AtomicMin(ref c.X0, x);
					AtomicMax(ref c.X1, x);
					AtomicMin(ref c.Y0, y);
					AtomicMax(ref c.Y1, y);

					for (var channel = 0; channel < sourceChannels; channel++)
					{
						unsafe
						{
							var srcColor = source[y * width * sourceChannels + x * sourceChannels + channel];
							Interlocked.Add(ref c.ChannelSum[channel], srcColor);
						}
					}
				}
			}

			count++;

			for (var ci = 0; ci < count; ci++)
			{
				ref var c = ref components[ci];
				
				// Add the first pixel of the component which is omitted
				var size = c.Size += 1;
				var swtSize = c.SwtSize += 1;
				var offset = ci * componentSizeLimit;
				var color = componentItems[offset] = c.Coloring;
				var s = componentSwtItems[offset] = swt[color];
				c.SwtSum += s;

				var w = c.Width = c.X1 - c.X0;
				var h = c.Height = c.Y1 - c.Y0;
				var d0 = c.MinDimension = Math.Min(w, h);
				var d1 = c.MaxDimension = Math.Max(w, h);
				c.SizeRatio = d1 / (float) d0;
				var diameter = c.Diameter = Hypot(w, h);
				var swtAverage = c.SwtAverage = c.SwtSum / (float)swtSize;

				Array.Sort(componentSwtItems, offset, swtSize);

				var cs = componentSwtItems.AsSpan(offset, swtSize);
				var swtMedian = c.SwtMedian = cs[swtSize >> 1];
				c.DiameterToSwtMedianRatio = diameter / swtMedian;
				
				var swtVariance = 0.0f;
				for (var si = 0; si < swtSize; si++)
				{
					var ei = cs[si] - swtAverage;
					swtVariance += ei * ei;
				}

				c.SwtVariance = swtVariance /= swtSize;

				for (var channel = 0; channel < sourceChannels; channel++)
				{
					unsafe
					{
						c.ChannelAverage[channel] = c.ChannelSum[channel] / size;
					}
				}
			}

			return count;
		}

		public static int TextDetectionFilter1(
			int count, 
			Component[] componentList,
			float varianceTolerance = 0.5f,
			float sizeRatioTolerance = 10)
		{
			var valid = 0;
			for (var ci = 0; ci < count; ci++)
			{
				ref var c = ref componentList[ci];
				
				var hasLowVariance = 
					c.SwtVariance < varianceTolerance * c.SwtAverage;
				var isSizeProportional = 
					c.SizeRatio < sizeRatioTolerance;
				var isSmall =
					c.Size < 10;

				if (true
					//isSmall 
					//   || hasLowVariance 
					//&& isSizeProportional
					)
					valid++;
				else
					c.Validity = -1;
			}

			return valid;
		}

		public static int DefaultFilter(
			int count,
			Component[] componentList,
			float varianceTolerance = 0.5f,
			float sizeRatioTolerance = 10)
		{
			var valid = 0;
			for (var ci = 0; ci < count; ci++)
			{
				ref var c = ref componentList[ci];

				var hasLowVariance =
					c.SwtVariance < varianceTolerance * c.SwtAverage;
				var isSizeProportional =
					c.SizeRatio < sizeRatioTolerance;
				var isDiameterSmall =
					c.DiameterToSwtMedianRatio < 10;


				if (
					hasLowVariance
					&& isSizeProportional
					&& isDiameterSmall
				)
					valid++;
				else
					c.Validity = -1;
			}

			return valid;
		}

		public static int PassiveFilter(
			int count,
			Component[] componentList,
			float varianceTolerance = 2.0f,
			float sizeRatioTolerance = 10)
		{
			var valid = 0;
			for (var ci = 0; ci < count; ci++)
			{
				ref var c = ref componentList[ci];

				var hasLowVariance =
					c.SwtVariance < varianceTolerance * c.SwtAverage;
				var isSizeProportional =
					c.SizeRatio < sizeRatioTolerance;
				var isDiameterSmall =
					c.DiameterToSwtMedianRatio < 15;

				if (!hasLowVariance)
					c.Validity |= Component.HasLowVariance;
				if (!isSizeProportional)
					c.Validity |= Component.IsSizeProportional;
				if (!isDiameterSmall)
					c.Validity |= Component.IsDiameterSmall;

				if (c.IsValid())
					valid++;
			}

			return valid;
		}

		public static RBush<Box2D> ComponentRBush(
			int count,
			Component[] componentList)
		{
			var rtree = new RBush<Box2D>();
			var boxes = new List<Box2D>(count);
			for (var ci = 0; ci < count; ci++)
			{
				ref var c = ref componentList[ci];
				if (!c.IsValid())
					continue;

				boxes.Add(c.BoundingBox(ci));
			}
			rtree.BulkLoad(boxes);
			return rtree;
		}

		public static void RemoveBoxes(
			int width,
			int height,
			int count,
			Component[] componentList,
			RBush<Box2D> rtree)
		{
			for (var ci = 0; ci < count; ci++)
			{
				ref var c0 = ref componentList[ci];
				if (!c0.IsValid())
					continue;

				var contains = 0;
				var box = c0.BoundingBox();
				var content = rtree.Search(box.Envelope).ToArray();
				foreach (var p in content)
				{
					if (ci == p.Data)
						continue;

					ref var c1 = ref componentList[p.Data];
					if (box.Contains(c1.BoundingBox()))
						contains++;
				}

				if (contains > 2)
				{
					c0.Validity = 0;
				}
			}
		}

		public static void MergeComponents(
			int width,
			int height,
			int count,
			Component[] componentList,
			RBush<Box2D> rtree)
		{
			for (var ci = 0; ci < count; ci++)
			{
				ref var c0 = ref componentList[ci];
				if (!c0.IsValid())
					continue;

				// Find all other component within a certain distance in horizontal direction
				var dim = Math.Max(Math.Max(c0.Width, c0.Height), 10);
				var nx0 = Math.Max(0, c0.X0 - dim);
				var nx1 = Math.Min(width - 1, c0.X1 + dim);
				var ny0 = Math.Max(0, c0.Y0 - dim);
				var ny1 = Math.Min(height - 1, c0.Y1 + dim);

				var near = rtree.Search(new Envelope(nx0, ny0, nx1, ny1));

				var join = 0;
				foreach (var p in near)
				{
					if (ci == p.Data)
						continue;

					ref var c1 = ref componentList[p.Data];

					// Horizontal overlap
					var dx = Math.Min(c0.X1, c1.X1) - Math.Max(c0.X0, c1.X0);
					if (dx <= 0)
						continue;

					// Vertical proximity
					var dv = c1.Height / 4;
					var c1y0 = Math.Max(0, c1.Y0 - dv);
					var c1y1 = Math.Min(height - 1, c1.Y1 + dv);
					var dy = Math.Min(c0.Y1, c1y1) - Math.Max(c0.Y0, c1y0);
					if (dy <= 0)
						continue;

					var mw = Math.Max(c0.Width, c1.Width);
					var mh = Math.Max(c0.Height, c1.Height);

					if (dx * 4 < mw || dy * 4 < mh)
						continue;

				}
			}
		}

		public static void AtomicMin(ref int location, int next)
		{
			while (true)
			{
				var current = location;
				next = Math.Min(current, next);
				if (current == next)
					return;

				var result = Interlocked.CompareExchange(ref location, next, current);
				if (result == current)
					return;
			}
		}

		public static void AtomicMax(ref int location, int next)
		{
			while (true)
			{
				var current = location;
				next = Math.Max(current, next);
				if (current == next)
					return;

				var result = Interlocked.CompareExchange(ref location, next, current);
				if (result == current)
					return;
			}
		}

		public static int Median(List<int> values, bool sort = true)
		{
			if (sort)
				values.Sort();

			var n = values.Count;
			var mid = n % 2 == 1 ? n / 2 : n / 2 - 1;
			if (mid < 0 ||mid >= values.Count)
				throw new ArgumentOutOfRangeException();
			return values[mid];
		}

		public static void Normalize(this float[] source, byte[] destination, int channels)
		{
			var min = new float[channels];
			var max = new float[channels];

			for (var c = 0; c < channels; c++)
			{
				min[c] = float.MaxValue;
				max[c] = float.MinValue;
			}

			for (var i = 0; i < source.Length; i++)
			{
				for (var c = 0; c < channels; c++)
				{
					var v = source[i + c];
					if (v < min[c])
						min[c] = v;
					if (v > max[c])
						max[c] = v;
				}
			}

			var len = new float[channels];
			for (var c = 0; c < channels; c++)
			{
				len[c] = max[c] - min[c];
			}

			for (var i = 0; i < source.Length; i++)
			{
				for (var c = 0; c < channels; c++)
				{
					var a = min[c];
					var l = len[c];
					var v = source[i + c];
					var r = ((v - a) / l) * 255;
					destination[i + c] = Convert.ToByte(r);
				}
			}
		}

		public static void Normalize(this int[] source, byte[] destination, int channels)
		{
			var min = new int[channels];
			var max = new int[channels];

			for (var c = 0; c < channels; c++)
			{
				min[c] = int.MaxValue;
				max[c] = int.MinValue;
			}

			for (var i = 0; i < source.Length; i++)
			{
				for (var c = 0; c < channels; c++)
				{
					var v = source[i + c];
					if (v < min[c])
						min[c] = v;
					if (v > max[c])
						max[c] = v;
				}
			}

			var len = new int[channels];
			for (var c = 0; c < channels; c++)
			{
				len[c] = max[c] - min[c];
			}

			for (var i = 0; i < source.Length; i++)
			{
				for (var c = 0; c < channels; c++)
				{
					var a = min[c];
					var l = len[c];
					var v = source[i + c];
					
					if (l == 0)
					{
						destination[i + c] = a > 0 ? (byte)255 : (byte)0;
					}
					else
					{
						var r = Math.Round((v - a) / (double) l) * 255;
						destination[i + c] = Convert.ToByte(r);
					}
				}
			}
		}

		private static int ClampToEdge(int i, int length)
		{
			if (i < 0)
				return 0;
			if (i >= length)
				return length - 1;
			return i;
		}

		private static int Clamp(int value, int max)
		{
			if (value == 0)
				return 1;
			if (value == max - 1)
				return max - 2;
			return value;
		}

		public static void Dump(this int[] data, int width, int height)
		{
			Console.WriteLine("{");
			for (var y = 0; y < height; y++)
			{
				for (var x = 0; x < width; x++)
				{
					var d = y * width + x;
					var v = data[d];
					Console.Write($"{(v == int.MaxValue ? 255 : v), 4}, ");
				}
				Console.WriteLine();
			}
			Console.WriteLine("}");
		}

		[Pure]
		public static byte[] ReplaceEquals(this byte[] data, byte value, byte result)
		{
			var copy = new byte[data.Length];
			Array.Copy(data, copy, data.Length);
			for (var i = 0; i < data.Length; i++)
			{
				if (copy[i] == value)
					copy[i] = result;
			}

			return copy;
		}

		[Pure]
		public static int[] ReplaceEquals(this int[] data, int value, int result)
		{
			var copy = new int[data.Length];
			Array.Copy(data, copy, data.Length);
			for (var i = 0; i < copy.Length; i++)
			{
				if (copy[i] == value)
					copy[i] = result;
			}

			return copy;
		}

		[Pure]
		public static float[] ReplaceEquals(this float[] data, float value, float result, float eps = 10e-6f)
		{
			var copy = new float[data.Length];
			Array.Copy(data, copy, data.Length);
			for (var i = 0; i < copy.Length; i++)
			{
				if (MathF.Abs(copy[i] - value) < eps)
					copy[i] = result;
			}

			return copy;
		}

		[Pure]
		public static byte[] ReplaceGreaterOrEquals(this byte[] data, byte value, byte result)
		{
			var copy = new byte[data.Length];
			Array.Copy(data, copy, data.Length);
			for (var i = 0; i < copy.Length; i++)
			{
				if (copy[i] >= value)
					copy[i] = result;
			}

			return copy;
		}

		[Pure]
		public static int[] ReplaceGreaterOrEquals(this int[] data, int value, int result)
		{
			var copy = new int[data.Length];
			Array.Copy(data, copy, data.Length);
			for (var i = 0; i < copy.Length; i++)
			{
				if (copy[i] >= value)
					copy[i] = result;
			}

			return copy;
		}

		[Pure]
		public static float[] ReplaceGreaterOrEquals(this float[] data, float value, float result)
		{
			var copy = new float[data.Length];
			Array.Copy(data, copy, data.Length);
			for (var i = 0; i < copy.Length; i++)
			{
				if (copy[i] >= value)
					copy[i] = result;
			}

			return copy;
		}

		[Pure]
		public static int[] MultiplyBy(this int[] data, int value)
		{
			var copy = new int[data.Length];
			Array.Copy(data, copy, data.Length);
			for (var i = 0; i < copy.Length; i++)
			{
				copy[i] *= value;
			}

			return copy;
		}
	}
}
