#include <QMessageBox>

#include "Model/Api/Validation.h"
#include "Gui/Settings.h"

#include "ComputationGridDialog.h"

ComputationGridDialog::ComputationGridDialog(SimulationConfig const& config, SimulationParameters const* parameters
	, QWidget * parent)
	: QDialog(parent), _parameters(parameters)
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

	connect(ui.gridSizeXEdit, &QLineEdit::textEdited, this, &ComputationGridDialog::updateUniverseSize);
	connect(ui.gridSizeYEdit, &QLineEdit::textEdited, this, &ComputationGridDialog::updateUniverseSize);
	connect(ui.unitSizeXEdit, &QLineEdit::textEdited, this, &ComputationGridDialog::updateUniverseSize);
	connect(ui.unitSizeYEdit, &QLineEdit::textEdited, this, &ComputationGridDialog::updateUniverseSize);
	connect(ui.buttonBox, &QDialogButtonBox::accepted, this, &ComputationGridDialog::okClicked);
}

optional<uint> ComputationGridDialog::getMaxThreads() const
{
	bool ok(true);
	double energy = ui.maxThreadsEdit->text().toUInt(&ok);
	if (!ok) {
		return boost::none;
	}
	return energy;
}

optional<IntVector2D> ComputationGridDialog::getGridSize() const
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

optional<IntVector2D> ComputationGridDialog::getUniverseSize() const
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

void ComputationGridDialog::updateUniverseSize()
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

void ComputationGridDialog::okClicked()
{
	optional<IntVector2D> universeSize = getUniverseSize();
	optional<IntVector2D> gridSize = getGridSize();
	optional<uint> maxThreads = getMaxThreads();
	if (!universeSize || !gridSize || !maxThreads) {
		QMessageBox msgBox(QMessageBox::Critical, "Error", "Wrong input.");
		return;
	}

	auto valResult = Validation::validate(*universeSize, *gridSize, _parameters);
	if (valResult == ValidationResult::Ok) {
		accept();
	}
	else if (valResult == ValidationResult::ErrorUnitSizeTooSmall) {
		QMessageBox msgBox(QMessageBox::Critical, "error", "Unit size is too small for simulation parameters.");
		msgBox.exec();
	}
	else {
		THROW_NOT_IMPLEMENTED();
	}
}

