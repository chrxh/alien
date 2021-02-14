#include <QTimer>

#include "MainController.h"
#include "InfoController.h"

InfoController::InfoController(QObject * parent)
	: QObject(parent)
	, _oneSecondTimer(new QTimer(this))
{
	connect(_oneSecondTimer, &QTimer::timeout, this, &InfoController::oneSecondTimerTimeout);
}

void InfoController::init(QLabel * infoLabel, MainController* mainController)
{
	_infoLabel = infoLabel;
	_mainController = mainController;
    _oneSecondTimer->stop();
    _oneSecondTimer->start(1000);
    _rendering = Rendering::Vector;
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

void InfoController::setRendering(Rendering value)
{
    _rendering = value;
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
	QString renderingString;
	if (Rendering::Pixel == _rendering) {
		renderingString = "Rendering: &nbsp;&nbsp;<font color=#FFB080><b>pixel graphic&nbsp;</b></font>";
	}
    else if (Rendering::Vector== _rendering) {
        renderingString = "Rendering: &nbsp;&nbsp;<font color=#B0FF80><b>vector graphic</b></font>";
    }
    else if (Rendering::Item == _rendering) {
        renderingString = "Rendering: &nbsp;&nbsp;<font color=#80B0FF><b>item graphic&nbsp;&nbsp;</b></font>";
    }
    else {
        THROW_NOT_IMPLEMENTED();
    }
    auto separator = QString("<br/>");  //QString("&nbsp;&nbsp;<font color=#7070FF>&#10072;</font>&nbsp;&nbsp;");
    auto infoString = renderingString
        + separator + QString("Zoom factor: %3x").arg(_zoomFactor)
        + separator + QString("Timestep: &nbsp;&nbsp;&nbsp;%1").arg(_mainController->getTimestep(), 9, 10, QLatin1Char('0'))
        + separator + QString("TPS: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;%2").arg(_tps, 5, 10, QLatin1Char('0'))
        + separator+ QString("&nbsp;");
	_infoLabel->setText(infoString);
}
