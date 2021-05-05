#include "AbstractWorldController.h"

#include <QGraphicsView>
#include <QScrollBar>

AbstractWorldController::AbstractWorldController(QGraphicsView* graphicsView, QObject* parent /*= nullptr*/)
    : QObject(parent)
    , _graphicsView(graphicsView)
{

}

