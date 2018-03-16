#pragma once

#include <QMessageBox>

#include "Model/Api/SimulationParameters.h"
#include "Gui/Definitions.h"

class SimulationParametersValidation
{
public:
	static bool validate(IntVector2D const& universeSize
		, IntVector2D const& gridSize
		, SimulationParameters const* parameters)
	{
		IntVector2D unitSize = { universeSize.x / gridSize.x, universeSize.y / gridSize.y };
		int range = std::min(unitSize.x, unitSize.y);
		if (parameters->cellFunctionCommunicatorRange < range
			&& parameters->cellFunctionSensorRange < range
			&& parameters->clusterMaxRadius < range)
		{
			return true;
		}
		QMessageBox msgBox(QMessageBox::Critical, "error", "Unit size is too small for simulation parameters.");
		msgBox.exec();
		return false;
	}
};
