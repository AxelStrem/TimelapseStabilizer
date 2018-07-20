#pragma once
#include "Frame.h"
#include <string>

class JPEGLoader
{
public:
	JPEGLoader();
	~JPEGLoader();

	static Frame LoadJPG(std::string path);
	static Frame LoadJPG(std::wstring path);

	static void SaveJPG(std::wstring path, const Frame& f, int quality = 100);
};

