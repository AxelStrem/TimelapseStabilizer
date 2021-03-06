// TimelapseStabilizer.cpp: определяет точку входа для консольного приложения.
//


#include <Windows.h>
#pragma comment( lib, "gdiplus.lib" ) 
#include <gdiplus.h> 

#define NOMINMAX

#include <Windows.h>
#include "Frame.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
// Gdiplus 

static const wchar_t* filename = L"D:\\From GoPro\\Bikes2\\100GOPRO\\G00";// "21090.jpg";
static const wchar_t* filename2 = L"D:\\From GoPro\\Bikes2\\100GOPRO\\G0021885.jpg";

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

class Filter
{
	std::vector<float> values;
	int head;
	float state;
public:
	Filter() : values(30), head(0), state(0.f) {}
	void push(float val)
	{
		state -= values[head];
		state += values[head++] = val;
		head %= values.size();
	}
	float value() const { return state/values.size(); }
};

struct FHeader
{
	int index;
	int xs;
	int ys;
	float rot;

	double rot_total;
	double xs_total;
	double ys_total;
};

int main_old(int argc, char* argv[])

{
	
	// Start Gdiplus 
	Gdiplus::GdiplusStartupInput gdiplusStartupInput;
	ULONG_PTR gdiplusToken;
	Gdiplus::GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);

	FrameStream stream;
	HDC hdc = GetDC(NULL);

	CLSID jpgClsid;
	GetEncoderClsid(L"image/jpeg", &jpgClsid);

	Gdiplus::EncoderParameters encoderParameters;
	encoderParameters.Count = 1;
	encoderParameters.Parameter[0].Guid = Gdiplus::EncoderQuality;
	encoderParameters.Parameter[0].Type = Gdiplus::EncoderParameterValueTypeLong;
	encoderParameters.Parameter[0].NumberOfValues = 1;

	ULONG quality = 50;
	encoderParameters.Parameter[0].Value = &quality;
	
	std::ifstream in("in.txt");
	if (in)
	{

		double xtot = 0.0;
		double ytot = 0.0;
		double rtot = 0.0;

		Filter xtot_filter;
		Filter ytot_filter;
		Filter rtot_filter;

		std::vector<FHeader> headers;

		while (true)
		{
			int i;
			int xr, yr;
			float rot;

			in >> i >> xr >> yr >> rot;

			if (in.eof())
				break;

			double as = sin(rtot);
			double ac = cos(rtot);
			rtot += rot;

			double nx = xr * ac - yr * as;
			double ny = xr * as + yr * ac;

			xtot += nx;
			ytot += ny;

			headers.push_back(FHeader{i,xr,yr,rot,rtot, xtot,ytot});
		}

		int index = 0;
		for (; index < 15; ++index)
		{
			xtot_filter.push(headers[index].xs_total);
			ytot_filter.push(headers[index].ys_total);
			rtot_filter.push(headers[index].rot_total);
		}

		std::wstring ffname(filename);
		ffname += std::to_wstring(headers.front().index) + L".jpg";

		Gdiplus::Bitmap* image_out = Gdiplus::Bitmap::FromFile(ffname.c_str());

		for(auto h : headers)
		{
			int i;
			int xr, yr;
			float rot;

			i = h.index;
			xr = h.xs;
			yr = h.ys;
			rot = h.rot;

			xtot = h.xs_total;
			ytot = h.ys_total;
			rtot = h.rot_total;

			std::wstring fname(filename);
			fname += std::to_wstring(i) + L".jpg";

			std::cout << "Frame #" << i << ": loading from " << std::string(fname.begin(), fname.end()) << std::endl;

			Gdiplus::Bitmap* image = Gdiplus::Bitmap::FromFile(fname.c_str());
			Frame f(image->GetWidth(), image->GetHeight(), [&image](int x, int y)
			{  Gdiplus::Color col;
			image->GetPixel(x, y, &col);
			return Pix(RGB(col.GetR(), col.GetG(), col.GetB()));
			});

			xtot_filter.push(headers[index].xs_total);
			ytot_filter.push(headers[index].ys_total);
			rtot_filter.push(headers[index].rot_total);

			index = std::min<int>((index + 1),headers.size()-1);

			FrameScaler fs(f);
			fs.x_shift = xtot - xtot_filter.value();
			fs.y_shift = ytot - ytot_filter.value();
			fs.SetRotation(rtot - rtot_filter.value());

			fs.traverse_precise([&](int x, int y, Pix color)
			{
				if(color.norm2()>0.0)
					image_out->SetPixel(x, y, Gdiplus::Color(color.RByte(), color.GByte(), color.BByte()));
			});

			std::wstring ofname(L"out\\");
			ofname += std::to_wstring(i) + L".jpg";
			image_out->Save(ofname.c_str(), &jpgClsid, &encoderParameters);
			delete image;
		}

		return 0;
	}

	if (argc < 3) return 0;

	std::string argv1(argv[1]);
	std::wstring fname_base(argv1.begin(), argv1.end());

	std::string argv2(argv[2]);
	std::istringstream iss(argv2);
	int start_index;
	iss >> start_index;

	for (int i = start_index; ; ++i)
	{
		std::wstring fname(fname_base);
		fname += std::to_wstring(i) + L".jpg";

		Gdiplus::Bitmap* image = Gdiplus::Bitmap::FromFile(fname.c_str());
		
		if (!image)
		{
			std::cout << "No file found";
			return 0;
		}

		Frame f(image->GetWidth(), image->GetHeight(), [&image](int x, int y)
		{  Gdiplus::Color col;
		image->GetPixel(x, y, &col);
		return Pix(RGB(col.GetR(), col.GetG(), col.GetB()));
		});

		delete image;

		std::ofstream out("out.txt",std::ios::app);

		stream.AddFrame(std::move(f));
		stream.process_last([&](int sx, int sy, float rot, const Frame& fr)
		{
			std::cout << "Frame #" << i - 21090 << ": " << sx << "x" << sy << "; @" << rot << "\r\n";
			FrameScaler fsc(fr.GetMipmap(2));
			fsc.SetRotation(rot);
			fsc.x_shift = sx / 4;
			fsc.y_shift = sy / 4;
			//fsc.traverse([&hdc](int x, int y, const Pix &p) { SetPixelV(hdc, x, y, p); });

			out << i << ' ' << sx << ' ' << sy << ' ' << rot << "\r\n";
		});

		out.close();
	}

	// Load the image 
	Gdiplus::Bitmap* image = Gdiplus::Bitmap::FromFile(filename);
	//image->
	// do something with your image 
	// ... 

	auto h = image->GetHeight();
	
	Frame f(image->GetWidth(), image->GetHeight(), [&image](int x, int y) 
		{  Gdiplus::Color col;
		   image->GetPixel(x, y, &col);
		   return Pix(RGB(col.GetR(), col.GetG(), col.GetB()));
		});

	Gdiplus::Bitmap* image2 = Gdiplus::Bitmap::FromFile(filename2);
	//image->
	// do something with your image 
	// ... 


	Frame f2(image2->GetWidth(), image2->GetHeight(), [&image2](int x, int y)
	{  Gdiplus::Color col;
	image2->GetPixel(x, y, &col);
	return Pix(RGB(col.GetR(), col.GetG(), col.GetB()));
	});

	
	const Frame& fm = f.GetMipmap(4);
	const Frame& fm2 = f2.GetMipmap(4);


	//while(true)
	fm.traverse([&hdc](int x, int y, const Pix &p) { SetPixelV(hdc, x, y, p); });


	FrameScaler fs(fm2);
	
	//fit_transform(fm, fs);

	while (true)
	{
		fm.traverse([&hdc](int x, int y, const Pix &p) { SetPixelV(hdc, x, y, p); });
		fs.traverse([&hdc](int x, int y, const Pix &p) { SetPixelV(hdc, x, y, p); });
	}

	//Gdiplus::Graphics g(GetDC(NULL));
	//g.DrawImage(image, 0, 0);
	

	// delete the image when done 
	delete image; image = 0;

	// Shutdown Gdiplus 
	Gdiplus::GdiplusShutdown(gdiplusToken);
}
