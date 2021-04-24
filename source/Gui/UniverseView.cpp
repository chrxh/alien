#include "UniverseView.h"

#include <QGraphicsView>
#include <QScrollBar>

UniverseView::UniverseView(QGraphicsView* graphicsView, QObject* parent /*= nullptr*/)
    : QObject(parent)
    , _graphicsView(graphicsView)
{

}

void UniverseView::centerToIntern(QVector2D const& scenePosition)
{
/*
    auto verticalScrollBarWidth =
        _graphicsView->verticalScrollBar()->isVisible() ? _graphicsView->verticalScrollBar()->width() : 0;
    auto horizontalScrollBarHeight =
        _graphicsView->horizontalScrollBar()->isVisible() ? _graphicsView->horizontalScrollBar()->height() : 0;
*/
/*
    disconnectView();
    _graphicsView->centerOn(scenePosition.x(), scenePosition.y());
    connectView();
*/
}
