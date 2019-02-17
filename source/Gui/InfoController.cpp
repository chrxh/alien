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

void InfoController::setDevice(Device value)
{
	_device = value;
}

void InfoController::oneSecondTimerTimeout()
{
	_tps = _tpsCounting;
	_tpsCounting = 0;
	updateInfoLabel();
}

void InfoController::updateInfoLabel()
{
	QString deviceString;
	if (Device::CPU == _device) {
		deviceString = "Device: <font color=#FF5050><b>CPU</b></font>";
	}
	if (Device::GPU == _device) {
		deviceString = "Device: <font color=#50FFFF><b>GPU</b></font>";
	}
	auto separator = QString("&nbsp;&nbsp;<font color=#7070FF>&#10072;</font>&nbsp;&nbsp;");
	auto infoString = deviceString + separator + QString("Timestep: %1").arg(_mainController->getTimestep(), 9, 10, QLatin1Char('0'))
		+ separator + QString("TPS: %2").arg(_tps, 5, 10, QLatin1Char('0'))
		+ separator + QString("Zoom factor: %3x").arg(_zoomFactor);
	_infoLabel->setText(infoString);
}
