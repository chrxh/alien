#include "startscreencontroller.h"
#include "gui/GuiSettings.h"

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
    turnOffScrollbar();
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
    _startScene->setBackgroundBrush(QBrush(BACKGROUND_COLOR));
    QPixmap logo("://tutorial/logo.png");
    _logoItem =_startScene->addPixmap(logo);
    _view->setMatrix(QMatrix());
    _view->scale(0.5, 0.5);
    _view->setScene(_startScene);
}

void StartScreenController::turnOffScrollbar ()
{
    _view->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    _view->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
}

void StartScreenController::timeout ()
{
    if( !isLogoTransparent() ) {
        scaleAndDecreaseOpacityOfLogo();
    }
    else {
        restoreScene();
        turnOnScrollbarAsNeeded();
        _timer->stop();
        Q_EMIT startScreenFinished();
    }
}

bool StartScreenController::isLogoTransparent () const
{
    return _logoItem->opacity() < 0.001;
}

void StartScreenController::scaleAndDecreaseOpacityOfLogo ()
{
    qreal scale = 1.0 + 1.0/LOGO_OPACITY_STEPS;
    _view->scale(scale, scale);
    qreal opacity = _logoItem->opacity();
    _logoItem->setOpacity(opacity - 1.0/LOGO_OPACITY_STEPS);
}

void StartScreenController::restoreScene ()
{
    _view->setScene(_savedScene);
    _view->setMatrix(_savedViewMatrix);
}

void StartScreenController::turnOnScrollbarAsNeeded ()
{
    _view->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    _view->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
}
