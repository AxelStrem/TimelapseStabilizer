#pragma once

#include <string>
#include <vector>

#include <filesystem>

#include "valtuple.hpp"
#include "Frame.h"

class FrameQueue
{
	std::wstring path;
	std::wstring out_path;

	std::filesystem::recursive_directory_iterator current_file_iterator;

	struct FrameData
	{
		int file_index;
		Frame frame;
		valtuple<float, 4> transform;

		FrameData() = default;
		FrameData(int fi, Frame&& f) : file_index(fi), frame(std::move(f)) {}
	};

	std::vector<FrameData> frames;
	valtuple<float, 4> combined_transform;
	int avg_count;
	int current_file;

	double xtot, ytot, rtot, ztot;
public:
	FrameQueue(std::wstring path);
	~FrameQueue();

	const Frame& LoadNextFrame();
	void  PushFrameTransform(float dx, float dy, float da, float dz);
	bool IsFrameReady();

	std::string GetFrameIndex() const;

	std::tuple<float, float, float, float> GetTopTransform();
	const Frame& GetTopFrame();
	void WriteAndPop(Frame&& f);
};

