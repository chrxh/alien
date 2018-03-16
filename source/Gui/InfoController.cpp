#include <QTimer>

#include "MainController.h"
#include "InfoController.h"

InfoController::InfoController(QObject * parent)
	: QObject(parent)
	, _oneSecondTimer(new QTimer(this))
{
	connect(_oneSecondTimer, &QTimer::timeout, this, &InfoController::oneSecondTimerTimeout);
	_oneSecondTimer->start(1000);
}

void InfoController::init(QLabel * infoLabel, MainController* mainController)
{
	_infoLabel = infoLabel;
	_mainController = mainController;
}

void InfoController::increaseTimestep()
{
	++_tpsCounting;
	updateInfoLabel();
}

void InfoController::setZoomFactor(double factor)
{
	_zoomFactor = factor;
	updateInfoLabel();
}

void InfoController::oneSecondTimerTimeout()
{
	_tps = _tpsCounting;
	_tpsCounting = 0;
	updateInfoLabel();
	static int i = 0;
}

void InfoController::updateInfoLabel()
{
	_infoLabel->setText(QString("Timestep: %1  TPS: %2  Zoom factor: %3x")
		.arg(_mainController->getTimestep(), 9, 10, QLatin1Char('0'))
		.arg(_tps, 5, 10, QLatin1Char('0'))
		.arg(_zoomFactor));
}
