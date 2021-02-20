#include <QMessageBox>

#include "Settings.h"
#include "StringHelper.h"
#include "ComputationSettingsDialog.h"
#include "SimulationConfig.h"

ComputationSettingsDialog::ComputationSettingsDialog(SimulationConfig const& config, QWidget * parent /*= nullptr*/)
	: QDialog(parent), _config(config)
{
	ui.setupUi(this);

    ui.computationSettingsWidget->setUniverseSize(config->universeSize);
    ui.computationSettingsWidget->setCudaConstants(config->cudaConstants);
    ui.extrapolateContentCheckBox->setChecked(
        GuiSettings::getSettingsValue(Const::ExtrapolateContentKey, Const::ExtrapolateContentDefault));

	connect(ui.buttonBox, &QDialogButtonBox::accepted, this, &ComputationSettingsDialog::okClicked);
}

optional<IntVector2D> ComputationSettingsDialog::getUniverseSize() const
{
	return ui.computationSettingsWidget->getUniverseSize();
}

optional<CudaConstants> ComputationSettingsDialog::getCudaConstants() const
{
    return ui.computationSettingsWidget->getCudaConstants();
}

optional<bool> ComputationSettingsDialog::isExtrapolateContent() const
{
    return ui.extrapolateContentCheckBox->isChecked();
}

void ComputationSettingsDialog::okClicked()
{
    auto const extrapolateContent = isExtrapolateContent();
    auto const size = getUniverseSize();
    auto const cudaConstants = getCudaConstants();
    if (!extrapolateContent || !size || !cudaConstants) {
        QMessageBox msgBox(QMessageBox::Critical, "Invalid values", Const::ErrorInvalidValues);
        msgBox.exec();
        return;
    }
    GuiSettings::setSettingsValue(Const::ExtrapolateContentKey, *isExtrapolateContent());
    ui.computationSettingsWidget->saveSettings();
    accept();
}

