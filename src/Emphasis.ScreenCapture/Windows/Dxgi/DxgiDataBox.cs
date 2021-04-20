﻿using System;
using System.Diagnostics.CodeAnalysis;
using System.Reactive.Disposables;

namespace Emphasis.ScreenCapture.Windows.Dxgi
{
	public class DxgiDataBox : IDisposable, ICancelable
	{
		public IntPtr DataPointer;
		public int RowPitch;
		public int SlicePitch;

		public DxgiDataBox(IntPtr dataPointer, int rowPitch, int slicePitch)
		{
			DataPointer = dataPointer;
			RowPitch = rowPitch;
			SlicePitch = slicePitch;
		}

		public bool IsEmpty => DataPointer == IntPtr.Zero && RowPitch == 0 && SlicePitch == 0;

		public bool IsDisposed => _disposable.IsDisposed;
		private readonly CompositeDisposable _disposable = new CompositeDisposable();

		public void Dispose()
		{
			_disposable.Dispose();
		}

		public void Add([NotNull] IDisposable disposable)
		{
			_disposable.Add(disposable);
		}

		public void Remove([NotNull] IDisposable disposable)
		{
			_disposable.Remove(disposable);
		}
	}
}