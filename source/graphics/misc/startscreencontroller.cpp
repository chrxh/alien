#include "startscreencontroller.h"

#include <QTimer>

StartScreenController::StartScreenController(QObject *parent)
    : QObject(parent), _startScene(new QGraphicsScene(this))
{

}

StartScreenController::~StartScreenController ()
{

}

void StartScreenController::runStartScreen (QGraphicsView* view)
{
    setupStartScreen(view);
    setupTimer();
}

void StartScreenController::setupStartScreen (QGraphicsView* view)
{
    _view = view;
    _scene = view->scene();
    _startScene->addRect(0.0, 0.0, 200.0, 100.0);
    view->setScene(_startScene);
}

void StartScreenController::setupTimer ()
{
    _timer = new QTimer(this);
    connect(_timer, SIGNAL(timeout()), this, SLOT(timeout()));
    _timer->start(3000);
}

void StartScreenController::timeout ()
{
    if( _view->scene() == _startScene ) {
        _view->setScene(_scene);
    }
    _timer->stop();
}
