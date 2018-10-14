#include <QMessageBox>

#include "Settings.h"
#include "StringHelper.h"
#include "ComputationSettingsDialog.h"
#include "SimulationConfig.h"

ComputationSettingsDialog::ComputationSettingsDialog(SimulationConfig const& config, QWidget * parent /*= nullptr*/)
	: QDialog(parent), _config(config)
{
	ui.setupUi(this);
	setFont(GuiSettings::getGlobalFont());

	if (auto configCpu = boost::dynamic_pointer_cast<_SimulationConfigCpu>(config)) {
		CHECK(configCpu->universeSize.x % configCpu->gridSize.x == 0);
		CHECK(configCpu->universeSize.y % configCpu->gridSize.y == 0);
		ui.gridSizeXEdit->setText(QString::number(configCpu->gridSize.x));
		ui.gridSizeYEdit->setText(QString::number(configCpu->gridSize.y));
		ui.maxThreadsEdit->setText(QString::number(configCpu->maxThreads));
		ui.unitSizeXEdit->setText(QString::number(configCpu->universeSize.x / configCpu->gridSize.x));
		ui.unitSizeYEdit->setText(QString::number(configCpu->universeSize.y / configCpu->gridSize.y));
	}
	ui.universeSizeXLabel->setText(QString::number(config->universeSize.x));
	ui.universeSizeYLabel->setText(QString::number(config->universeSize.y));

	connect(ui.gridSizeXEdit, &QLineEdit::textEdited, this, &ComputationSettingsDialog::updateLabels);
	connect(ui.gridSizeYEdit, &QLineEdit::textEdited, this, &ComputationSettingsDialog::updateLabels);
	connect(ui.unitSizeXEdit, &QLineEdit::textEdited, this, &ComputationSettingsDialog::updateLabels);
	connect(ui.unitSizeYEdit, &QLineEdit::textEdited, this, &ComputationSettingsDialog::updateLabels);
	connect(ui.maxThreadsEdit, &QLineEdit::textEdited, this, &ComputationSettingsDialog::updateLabels);
	connect(ui.buttonBox, &QDialogButtonBox::accepted, this, &ComputationSettingsDialog::okClicked);
}

optional<uint> ComputationSettingsDialog::getMaxThreads() const
{
	bool ok(true);
	double energy = ui.maxThreadsEdit->text().toUInt(&ok);
	if (!ok) {
		return boost::none;
	}
	return energy;
}

optional<IntVector2D> ComputationSettingsDialog::getGridSize() const
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

optional<IntVector2D> ComputationSettingsDialog::getUniverseSize() const
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

void ComputationSettingsDialog::updateLabels()
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

	int limitThreads = *getMaxThreads();
	int activeThreads = std::min((gridSizeX / 3) * (gridSizeY / 3), limitThreads);
	int totalThreads = gridSizeX * gridSizeY;
	ui.activeThreadsLabel->setText(StringHelper::toString(activeThreads) + QString(" (active)"));
	ui.totalThreadsLabel->setText(StringHelper::toString(totalThreads) + QString(" (total)"));

}

void ComputationSettingsDialog::okClicked()
{
	optional<IntVector2D> universeSize = getUniverseSize();
	optional<IntVector2D> gridSize = getGridSize();
	optional<uint> maxThreads = getMaxThreads();
	if (!universeSize || !gridSize || !maxThreads) {
		QMessageBox msgBox(QMessageBox::Critical, "Error", "Wrong input.");
		return;
	}

	_config->universeSize = *universeSize;
	if (auto configCpu = boost::dynamic_pointer_cast<_SimulationConfigCpu>(_config)) {
		configCpu->gridSize = *gridSize;
		configCpu->maxThreads = *maxThreads;
	}
	string errorMsg;
	auto valResult = _config->validate(errorMsg);
	if (valResult == _SimulationConfig::ValidationResult::Ok) {
		accept();
	}
	else if (valResult == _SimulationConfig::ValidationResult::Error) {
		QMessageBox msgBox(QMessageBox::Critical, "error", errorMsg.c_str());
		msgBox.exec();
	}
	else {
		THROW_NOT_IMPLEMENTED();
	}
}

