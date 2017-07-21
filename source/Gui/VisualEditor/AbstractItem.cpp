#include "AbstractItem.h"

#include "Gui/Settings.h"

void AbstractItem::move(QVector2D const & delta)
{
	QGraphicsItem::moveBy(delta.x(), delta.y());
}
