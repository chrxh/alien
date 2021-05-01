#include "UniverseView.h"

#include <QGraphicsView>
#include <QScrollBar>

UniverseView::UniverseView(QGraphicsView* graphicsView, QObject* parent /*= nullptr*/)
    : QObject(parent)
    , _graphicsView(graphicsView)
{

}

