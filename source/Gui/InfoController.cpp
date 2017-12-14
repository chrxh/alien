#include <QTimer>

#include "InfoController.h"

InfoController::InfoController(QObject * parent)
	: QObject(parent)
	, _oneSecondTimer(new QTimer(this))
{
	connect(_oneSecondTimer, &QTimer::timeout, this, &InfoController::oneSecondTimerTimeout);
	_oneSecondTimer->start(1000);
}

void InfoController::init(QLabel * infoLabel)
{
	_infoLabel = infoLabel;
}

void InfoController::setTimestep(int timestep)
{
	_timestep = 0;
	_tps = 0;
	_tpsCounting = 0;
	updateInfoLabel();
}

void InfoController::increaseTimestep()
{
	++_tpsCounting;
	++_timestep;
	updateInfoLabel();
}

void InfoController::oneSecondTimerTimeout()
{
	_tps = _tpsCounting;
	_tpsCounting = 0;
	updateInfoLabel();
}

void InfoController::updateInfoLabel()
{
	_infoLabel->setText(QString("Timestep: %1  TPS: %2  Magnification: %3x")
		.arg(_timestep, 9, 10, QLatin1Char('0'))
		.arg(_tps, 5, 10, QLatin1Char('0'))
		.arg(_magnification));
}
