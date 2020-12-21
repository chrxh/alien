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

    ui.computationSettingsWidget->setUniverseSize(config->universeSize);
    ui.computationSettingsWidget->setCudaConstants(config->cudaConstants);
    ui.extrapolateContentCheckBox->setChecked(
        GuiSettings::getSettingsValue(Const::ExtrapolateContentKey, Const::ExtrapolateContentDefault));

	connect(ui.buttonBox, &QDialogButtonBox::accepted, this, &ComputationSettingsDialog::okClicked);
}

IntVector2D ComputationSettingsDialog::getUniverseSize() const
{
	return ui.computationSettingsWidget->getUniverseSize();
}

CudaConstants ComputationSettingsDialog::getCudaConstants() const
{
    return ui.computationSettingsWidget->getCudaConstants();
}

bool ComputationSettingsDialog::isExtrapolateContent() const
{
    return ui.extrapolateContentCheckBox->isChecked();
}

void ComputationSettingsDialog::okClicked()
{
    GuiSettings::setSettingsValue(Const::ExtrapolateContentKey, isExtrapolateContent());
    accept();
}

