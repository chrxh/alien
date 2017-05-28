#ifndef UNITOBSERVER_H
#define UNITOBSERVER_H

#include "Model/Definitions.h"

class UnitObserver
{
public:
	virtual ~UnitObserver() = default;

	virtual void unregister() = 0;
	virtual void accessToUnits() = 0;
};

#endif // UNITOBSERVER_H
