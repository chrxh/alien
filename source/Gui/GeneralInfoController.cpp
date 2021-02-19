#include <QTimer>

#include "MainController.h"
#include "SimulationConfig.h"
#include "GeneralInfoController.h"
#include "Settings.h"
#include "StringHelper.h"

GeneralInfoController::GeneralInfoController(QObject * parent)
	: QObject(parent)
	, _oneSecondTimer(new QTimer(this))
{
	connect(_oneSecondTimer, &QTimer::timeout, this, &GeneralInfoController::oneSecondTimerTimeout);
}

void GeneralInfoController::init(QLabel * infoLabel, MainController* mainController)
{
	_infoLabel = infoLabel;
	_mainController = mainController;
    _oneSecondTimer->stop();
    _oneSecondTimer->start(1000);
    _rendering = Rendering::Vector;
}

void GeneralInfoController::increaseTimestep()
{
	++_tpsCounting;
	updateInfoLabel();
}

void GeneralInfoController::setZoomFactor(double factor)
{
	_zoomFactor = factor;
	updateInfoLabel();
}

void GeneralInfoController::setRendering(Rendering value)
{
    _rendering = value;
    updateInfoLabel();
}

void GeneralInfoController::oneSecondTimerTimeout()
{
	_tps = _tpsCounting;
	_tpsCounting = 0;
	updateInfoLabel();
}

void GeneralInfoController::updateInfoLabel()
{
    auto config = _mainController->getSimulationConfig();
    if (!config) {
        return;
    }

    auto renderModeValueString = [&] {
        if (Rendering::Pixel == _rendering) {
            return QString("pixel");
        } else if (Rendering::Vector == _rendering) {
            return QString("vector");
        }
        return QString("item");
    }();
    auto worldSizeValueString = QString("%1 x %2").arg(
        StringHelper::generateFormattedIntString(config->universeSize.x, true),
        StringHelper::generateFormattedIntString(config->universeSize.y, true));
    auto zoomLevelValueString = QString("%1x").arg(_zoomFactor);
    auto timestepValueString = StringHelper::generateFormattedIntString(_mainController->getTimestep(), true);
    auto tpsValueString = StringHelper::generateFormattedIntString(_tps, true);

    auto renderModeColorString = [&] {
        if (Rendering::Pixel == _rendering) {
            return QString("<font color = #FFB080>");
        } else if (Rendering::Vector == _rendering) {
            return QString("<font color = #B0FF80>");
        }
        return QString("<font color = #80B0FF>");
    }();
    auto colorTextStart = QString("<font color = %1>").arg(Const::CellEditTextColor1.name());
    auto colorDataStart = QString("<font color = %1>").arg(Const::CellEditDataColor1.name());
    auto colorEnd = QString("</font>");

    QString renderingString = colorTextStart + "Render style: " + colorEnd + renderModeColorString + "<b>"
        + renderModeValueString + "</b>" + colorEnd;
    QString worldSizeString = colorTextStart + "World size: &nbsp;&nbsp;" + colorEnd + colorDataStart + "<b>"
        + worldSizeValueString + "</b>" + colorEnd;
    QString zoomLevelString = colorTextStart + "Zoom level: &nbsp;&nbsp;" + colorEnd + colorDataStart + "<b>"
        + zoomLevelValueString + "</b>" + colorEnd;
    QString timestepString = colorTextStart + "Time step: &nbsp;&nbsp;&nbsp;" + colorEnd + colorDataStart + "<b>"
        + timestepValueString + "</b>" + colorEnd;
    QString tpsString = colorTextStart + "Time steps/s: " + colorEnd + colorDataStart + "<b>"
        + tpsValueString + "</b>" + colorEnd;

    auto separator = QString("<br/>");  
    auto infoString = renderingString + separator + worldSizeString + separator + zoomLevelString + separator
        + timestepString + separator + tpsString;
	_infoLabel->setText(infoString);
}
