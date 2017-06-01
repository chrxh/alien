#pragma once

class GpuObserver
{
public:
	virtual ~GpuObserver() = default;

	virtual void unregister() = 0;
	virtual void accessToUnits() = 0;
};
