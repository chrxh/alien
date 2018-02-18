
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

	connect(ui.gridSizeXEdit, &QLineEdit::textEdited, this, &SimulationConfigDialog::updateUniverseSize);
	connect(ui.gridSizeYEdit, &QLineEdit::textEdited, this, &SimulationConfigDialog::updateUniverseSize);
	connect(ui.unitSizeXEdit, &QLineEdit::textEdited, this, &SimulationConfigDialog::updateUniverseSize);
	connect(ui.unitSizeYEdit, &QLineEdit::textEdited, this, &SimulationConfigDialog::updateUniverseSize);
}

optional<uint> SimulationConfigDialog::getMaxThreads() const
{
	bool ok(true);
	double energy = ui.maxThreadsEdit->text().toUInt(&ok);
	if (!ok) {
		return boost::none;
	}
	return energy;
}

optional<IntVector2D> SimulationConfigDialog::getGridSize() const
{
	IntVector2D result;
	bool ok(true);
	result.x = ui.gridSizeXEdit->text().toUInt(&ok);
	if (!ok) {
		return boost::none;
	}
	result.y = ui.gridSizeYEdit->text().toUInt(&ok);
	if (!ok) {
		return boost::none;
	}

	return result;
}

optional<IntVector2D> SimulationConfigDialog::getUniverseSize() const
{
	IntVector2D result;
	bool ok(true);
	result.x = ui.universeSizeXLabel->text().toUInt(&ok);
	if (!ok) {
		return boost::none;
	}
	result.y = ui.universeSizeYLabel->text().toUInt(&ok);
	if (!ok) {
		return boost::none;
	}

	return result;
}

void SimulationConfigDialog::updateUniverseSize()
{
	bool ok = false;
	int gridSizeX = ui.gridSizeXEdit->text().toUInt(&ok);
	if (!ok) { return; }

	int gridSizeY = ui.gridSizeYEdit->text().toUInt(&ok);
	if (!ok) { return; }

	int unitSizeX = ui.unitSizeXEdit->text().toUInt(&ok);
	if (!ok) { return; }

	int unitSizeY = ui.unitSizeYEdit->text().toUInt(&ok);
	if (!ok) { return; }

	IntVector2D universeSize = { gridSizeX * unitSizeX, gridSizeY * unitSizeY };
	ui.universeSizeXLabel->setText(QString::fromStdString(std::to_string(universeSize.x)));
	ui.universeSizeYLabel->setText(QString::fromStdString(std::to_string(universeSize.y)));
}

