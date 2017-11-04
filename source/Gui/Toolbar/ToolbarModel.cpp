#include "Gui/Definitions.h"
#include "ToolbarModel.h"


QVector2D ToolbarModel::getPositionDeltaForNewEntity()
{
	_delta += 1.0;
	if (_delta > 10.0) {
		_delta = 0.0;
	}
	return QVector2D(_delta, -_delta);
}
