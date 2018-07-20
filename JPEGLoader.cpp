#include "JPEGLoader.h"
#include <Windows.h>

#include <iostream>

#pragma comment( lib, "gdiplus.lib" ) 
#include <gdiplus.h> 


int GetEncoderClsid(const WCHAR* format, CLSID* pClsid)
{
	UINT  num = 0;          // number of image encoders
	UINT  size = 0;         // size of the image encoder array in bytes

	Gdiplus::ImageCodecInfo* pImageCodecInfo = NULL;

	Gdiplus::GetImageEncodersSize(&num, &size);
	if (size == 0)
		return -1;  // Failure

	pImageCodecInfo = (Gdiplus::ImageCodecInfo*)(malloc(size));
	if (pImageCodecInfo == NULL)
		return -1;  // Failure

	GetImageEncoders(num, size, pImageCodecInfo);

	for (UINT j = 0; j < num; ++j)
	{
		if (wcscmp(pImageCodecInfo[j].MimeType, format) == 0)
		{
			*pClsid = pImageCodecInfo[j].Clsid;
			free(pImageCodecInfo);
			return j;  // Success
		}
	}

	free(pImageCodecInfo);
	return -1;  // Failure
}


struct JPEG_Preset
{
	Gdiplus::GdiplusStartupInput gdiplusStartupInput;
	ULONG_PTR gdiplusToken;
	CLSID jpgClsid;
	ULONG quality = 50;

	JPEG_Preset()
	{
		Gdiplus::GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);
		
		GetEncoderClsid(L"image/jpeg", &jpgClsid);

		Gdiplus::EncoderParameters encoderParameters;
		encoderParameters.Count = 1;
		encoderParameters.Parameter[0].Guid = Gdiplus::EncoderQuality;
		encoderParameters.Parameter[0].Type = Gdiplus::EncoderParameterValueTypeLong;
		encoderParameters.Parameter[0].NumberOfValues = 1;

		encoderParameters.Parameter[0].Value = &quality;
	}

	~JPEG_Preset()
	{
		Gdiplus::GdiplusShutdown(gdiplusToken);
	}
};

JPEGLoader::JPEGLoader()
{
}


JPEGLoader::~JPEGLoader()
{
}

Frame JPEGLoader::LoadJPG(std::string path)
{
	std::wstring wp(path.begin(), path.end());
	return LoadJPG(wp);
}

Frame JPEGLoader::LoadJPG(std::wstring path)
{
	static JPEG_Preset jpgp;
	
	Gdiplus::Bitmap* image = Gdiplus::Bitmap::FromFile(path.c_str());


	auto x = path.find(L"Bikes");

	Frame f;


	Gdiplus::BitmapData bdata;
	image->LockBits(nullptr, 0, PixelFormat24bppRGB, &bdata);
	
	//if (x < path.size())
	{
		f = std::move(Frame(bdata.Width, bdata.Height, [&bdata](int x, int y)
		{  Gdiplus::Color col;
		unsigned char* p = reinterpret_cast<unsigned char*>(bdata.Scan0) + (bdata.Stride*(bdata.Height-y-1) + 3 * x);
		return Pix(RGB(p[2], p[1], p[0]));
		}));
	}
	/*else
	{

		f = std::move(Frame(bdata.Width, bdata.Height, [&bdata](int x, int y)
		{  Gdiplus::Color col;
		unsigned char* p = reinterpret_cast<unsigned char*>(bdata.Scan0) + (bdata.Stride*y + 3 * (bdata.Width - x - 1));
		return Pix(RGB(p[2], p[1], p[0]));
		}));
	}*/


	image->UnlockBits(&bdata);

	delete image;

	return f;
}

void JPEGLoader::SaveJPG(std::wstring path, const Frame & f, int quality)
{
	Gdiplus::Bitmap image(f.GetWidth(), f.GetHeight(), PixelFormat24bppRGB);
	Gdiplus::BitmapData bdata;

	CLSID jpgClsid;
	GetEncoderClsid(L"image/jpeg", &jpgClsid);

	Gdiplus::EncoderParameters encoderParameters;
	encoderParameters.Count = 1;
	encoderParameters.Parameter[0].Guid = Gdiplus::EncoderQuality;
	encoderParameters.Parameter[0].Type = Gdiplus::EncoderParameterValueTypeLong;
	encoderParameters.Parameter[0].NumberOfValues = 1;

	ULONG q = quality;
	encoderParameters.Parameter[0].Value = &q;

	image.LockBits(nullptr, 0, PixelFormat24bppRGB, &bdata);
	f.traverse([&](int x, int y, Pix p)
	{
		unsigned char* c = reinterpret_cast<unsigned char*>(bdata.Scan0) + (bdata.Stride*(bdata.Height - y - 1) + 3 * x);
		c[0] = static_cast<unsigned char>(p.b*255.f);
		c[1] = static_cast<unsigned char>(p.g*255.f);
		c[2] = static_cast<unsigned char>(p.r*255.f);
	});
	image.UnlockBits(&bdata);

	image.Save(path.c_str(), &jpgClsid, &encoderParameters);

}
