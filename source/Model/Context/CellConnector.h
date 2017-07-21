#pragma once

#include "Model/Entities/Descriptions.h"

class CellConnector
{
public:
	virtual ~CellConnector() = default;

	virtual void update(DataDescription &data) const = 0;
};

