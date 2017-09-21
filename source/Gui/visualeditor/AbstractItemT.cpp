#include "AbstractItemT.h"

#include "Gui/SettingsT.h"

void AbstractItem::moveBy(QVector2D const & delta)
{
	QGraphicsItem::moveBy(delta.x(), delta.y());
}
