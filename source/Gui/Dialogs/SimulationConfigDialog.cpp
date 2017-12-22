
#include "Gui/Settings.h"
#include "SimulationConfigDialog.h"

SimulationConfigDialog::SimulationConfigDialog(SimulationConfig const& config, QWidget * parent)
	: QDialog(parent)
{
	ui.setupUi(this);
	setFont(GuiSettings::getGlobalFont());

	CHECK(config.universeSize.x % config.gridSize.x == 0);
	CHECK(config.universeSize.y % config.gridSize.y == 0);
	ui.gridSizeXEdit->setText(QString::number(config.gridSize.x));
	ui.gridSizeYEdit->setText(QString::number(config.gridSize.y));
	ui.unitSizeXEdit->setText(QString::number(config.universeSize.x / config.gridSize.x));
	ui.unitSizeYEdit->setText(QString::number(config.universeSize.y / config.gridSize.y));
	ui.universeSizeXLabel->setText(QString::number(config.universeSize.x));
	ui.universeSizeYLabel->setText(QString::number(config.universeSize.y));
	ui.maxThreadsEdit->setText(QString::number(config.maxThreads));
}
