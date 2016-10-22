#include "startscreencontroller.h"

#include <QTimer>
#include <QGraphicsItem>

const int LOGO_DURATION_MS = 3000;
const int LOGO_OPACITY_STEPS = 100;

StartScreenController::StartScreenController(QObject *parent)
    : QObject(parent), _startScene(new QGraphicsScene(this))
{

}

StartScreenController::~StartScreenController ()
{

}

void StartScreenController::runStartScreen (QGraphicsView* view)
{
    setupStartScene(view);
    setupTimer();
}

void StartScreenController::setupStartScene (QGraphicsView* view)
{
    saveSceneAndView(view);
    createSceneWithLogo();
}

void StartScreenController::setupTimer ()
{
    _timer = new QTimer(this);
    connect(_timer, SIGNAL(timeout()), this, SLOT(timeout()));
    _timer->start(LOGO_DURATION_MS/LOGO_OPACITY_STEPS);
}

void StartScreenController::saveSceneAndView (QGraphicsView* view)
{
    _view = view;
    _savedScene = view->scene();
    _savedViewMatrix = _view->matrix();
}

void StartScreenController::createSceneWithLogo ()
{
    _startScene->setBackgroundBrush(QBrush(QColor(0,0,0)));
    QPixmap logo("://tutorial/logo.png");
    _logoItem =_startScene->addPixmap(logo);
    _view->setMatrix(QMatrix());
    _view->scale(0.75, 0.75);
    _view->setScene(_startScene);
}

void StartScreenController::timeout ()
{
    decreaseOpacityOfLogo();
    if( isLogoTransparent() ) {
        restoreScene();
        _timer->stop();
    }
}

void StartScreenController::decreaseOpacityOfLogo ()
{
    qreal opacity = _logoItem->opacity();
    if( !isLogoTransparent() ) {
        _logoItem->setOpacity(opacity - 1.0/LOGO_OPACITY_STEPS);
    }
}

bool StartScreenController::isLogoTransparent () const
{
    return _logoItem->opacity() < 0.001;
}

void StartScreenController::restoreScene ()
{
    if( _view->scene() == _startScene ) {
        _view->setScene(_savedScene);
        _view->setMatrix(_savedViewMatrix);
    }
}
