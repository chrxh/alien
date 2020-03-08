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

    auto const configGpu = boost::static_pointer_cast<_SimulationConfigGpu>(config);
    ui.computationSettingsWidget->setUniverseSize(config->universeSize);
    ui.computationSettingsWidget->setNumBlocks(configGpu->numBlocks);
    ui.computationSettingsWidget->setNumThreadsPerBlock(configGpu->numThreadsPerBlock);
    ui.computationSettingsWidget->setMaxClusters(configGpu->maxClusters);
    ui.computationSettingsWidget->setMaxCells(configGpu->maxCells);
    ui.computationSettingsWidget->setMaxTokens(configGpu->maxTokens);
    ui.computationSettingsWidget->setMaxParticles(configGpu->maxParticles);
    ui.computationSettingsWidget->setDynamicMemorySize(configGpu->dynamicMemorySize);
    ui.extrapolateContentCheckBox->setChecked(
        GuiSettings::getSettingsValue(Const::ExtrapolateContentKey, Const::ExtrapolateContentDefault));

	connect(ui.buttonBox, &QDialogButtonBox::accepted, this, &ComputationSettingsDialog::okClicked);
}

IntVector2D ComputationSettingsDialog::getUniverseSize() const
{
	return ui.computationSettingsWidget->getUniverseSize();
}

uint ComputationSettingsDialog::getNumBlocks() const
{
    return ui.computationSettingsWidget->getNumBlocks();
}

uint ComputationSettingsDialog::getNumThreadsPerBlock() const
{
    return ui.computationSettingsWidget->getNumThreadsPerBlock();
}

uint ComputationSettingsDialog::getMaxClusters() const
{
    return ui.computationSettingsWidget->getMaxClusters();
}

uint ComputationSettingsDialog::getMaxCells() const
{
    return ui.computationSettingsWidget->getMaxCells();
}

uint ComputationSettingsDialog::getMaxTokens() const
{
    return ui.computationSettingsWidget->getMaxTokens();
}

uint ComputationSettingsDialog::getMaxParticles() const
{
    return ui.computationSettingsWidget->getMaxParticles();
}

uint ComputationSettingsDialog::getDynamicMemorySize() const
{
    return ui.computationSettingsWidget->getDynamicMemorySize();
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

