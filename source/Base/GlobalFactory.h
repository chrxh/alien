#pragma once

#include "Definitions.h"

class GlobalFactory
{
public:
	virtual ~GlobalFactory() = default;

	virtual NumberGenerator* buildRandomNumberGenerator() const = 0;
};
