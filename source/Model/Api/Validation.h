#pragma once

#include "Definitions.h"

enum class ValidationResult
{
	Ok,
	ErrorUnitSizeTooSmall
};

class MODEL_EXPORT Validation
{
public:

	static ValidationResult validate(IntVector2D const& universeSize
		, IntVector2D const& gridSize
		, SimulationParameters const* parameters);

};
