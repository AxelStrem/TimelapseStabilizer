#include "FrameQueue.h"
#include "JPEGLoader.h"

#include <filesystem>
#include <iostream>
#include <algorithm>

FrameQueue::FrameQueue(std::wstring p) : path(p), avg_count(15), frames(15), current_file_iterator(p)
{
	current_file = 0;
	out_path = L"out\\";
	xtot = 0.f;
	ytot = 0.f;
	ztot = 0.f;
	rtot = 0.f;

	for (int i = 0; i < 1405; ++i)
	{
		current_file_iterator++;
	}
}


FrameQueue::~FrameQueue()
{
}

const Frame & FrameQueue::LoadNextFrame()
{
	//std::wstring filename = path + std::to_wstring(current_file) + L".JPG";
	std::string filename;
	do
	{
		filename = current_file_iterator->path().string();
		std::transform(filename.begin(), filename.end(), filename.begin(), ::tolower);
		current_file_iterator++;
	} while (filename.find(".jpg") == filename.npos);

	frames.emplace_back(current_file++, JPEGLoader::LoadJPG(filename));
	return frames.back().frame;
}

void FrameQueue::PushFrameTransform(float dx, float dy, float da, float dz)
{
	double as = sin(rtot);
	double ac = cos(rtot);
	rtot += da;

	double nx = dx * ac - dy * as;
	double ny = dx * as + dy * ac;

	xtot += nx;
	ytot += ny;
	ztot += dz;

	auto v = valtuple<float, 4>{ { (float)xtot,(float)ytot,(float)rtot,(float)ztot } };
	(frames.end() - 1)->transform = v;
	avg_count++;
	combined_transform = combined_transform + v;
}

bool FrameQueue::IsFrameReady()
{
	return avg_count>=30;
}

std::string FrameQueue::GetFrameIndex() const
{
	return current_file_iterator->path().string();
}

std::tuple<float, float, float, float> FrameQueue::GetTopTransform()
{
	auto v = frames[15].transform;
	auto avg = combined_transform / static_cast<float>(avg_count);
	v = v - avg;
	return std::tuple<float, float, float, float>(v[0],v[1],v[2],v[3]);
}

const Frame & FrameQueue::GetTopFrame()
{
	return frames[15].frame;
}

void FrameQueue::WriteAndPop(Frame && f)
{
	std::wstring filename = out_path + L"Z" + std::to_wstring(current_file) + L".JPG";

	frames[15].frame = std::move(f);
	avg_count--;
	combined_transform = combined_transform - frames.front().transform;

	JPEGLoader::SaveJPG(filename, frames[15].frame);

	frames[15].frame.Clear();
	frames.erase(frames.begin());
}
