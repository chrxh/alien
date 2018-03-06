#pragma once

class UnitObserver
{
public:
	virtual ~UnitObserver() = default;

	virtual void unregister() = 0;
	virtual void accessToUnits() = 0;
};

