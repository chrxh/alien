#include "AbstractItem.h"

#include "Gui/Settings.h"

void AbstractItem::moveBy(QVector2D const & delta)
{
	QGraphicsItem::moveBy(delta.x(), delta.y());
	updateDescription();
}
