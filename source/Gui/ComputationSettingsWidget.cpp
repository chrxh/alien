#include "ComputationSettingsWidget.h"
#include "Settings.h"
#include "StringHelper.h"

namespace
{
    uint getUIntOrZero(QString const& string)
    {
        bool ok(true);
        auto const value = string.toUInt(&ok);
        if (!ok) {
            return 0;
        }
        return value;
    }
}

ComputationSettingsWidget::ComputationSettingsWidget(QWidget* parent)
    : QWidget(parent)
{
    ui.setupUi(this);

    setFont(GuiSettings::getGlobalFont());
    ui.gpuUniverseSizeXEdit->setText(StringHelper::toString(
        GuiSettings::getSettingsValue(Const::GpuUniverseSizeXKey, Const::GpuUniverseSizeXDefault)));
    ui.gpuUniverseSizeYEdit->setText(StringHelper::toString(
        GuiSettings::getSettingsValue(Const::GpuUniverseSizeYKey, Const::GpuUniverseSizeYDefault)));
    ui.gpuNumBlocksEdit->setText(StringHelper::toString(
        GuiSettings::getSettingsValue(Const::GpuNumBlocksKey, Const::GpuNumBlocksDefault)));
    ui.gpuNumThreadsPerBlockEdit->setText(StringHelper::toString(
        GuiSettings::getSettingsValue(Const::GpuNumThreadsPerBlockKey, Const::GpuNumThreadsPerBlockDefault)));
    ui.gpuMaxClustersEdit->setText(StringHelper::toString(
        GuiSettings::getSettingsValue(Const::GpuMaxClustersKey, Const::GpuMaxClustersDefault)));
    ui.gpuMaxCellsEdit->setText(StringHelper::toString(
        GuiSettings::getSettingsValue(Const::GpuMaxCellsKey, Const::GpuMaxCellsDefault)));
    ui.gpuMaxTokensEdit->setText(StringHelper::toString(
        GuiSettings::getSettingsValue(Const::GpuMaxTokensKey, Const::GpuMaxTokensDefault)));
    ui.gpuMaxParticlesEdit->setText(StringHelper::toString(
        GuiSettings::getSettingsValue(Const::GpuMaxParticlesKey, Const::GpuMaxParticlesDefault)));
    ui.gpuDynamicMemorySizeEdit->setText(StringHelper::toString(
        GuiSettings::getSettingsValue(Const::GpuDynamicMemorySizeKey, Const::GpuDynamicMemorySizeDefault)));
}

IntVector2D ComputationSettingsWidget::getUniverseSize() const
{
    IntVector2D result;
    bool ok;
    result.x = ui.gpuUniverseSizeXEdit->text().toUInt(&ok);
    if (!ok) {
        result.x = 100;
    }

    result.y = ui.gpuUniverseSizeYEdit->text().toUInt(&ok);
    if (!ok) {
        result.y = 100;
    }

    return result;
}

void ComputationSettingsWidget::setUniverseSize(IntVector2D const & value) const
{
    ui.gpuUniverseSizeXEdit->setText(QString::number(value.x));
    ui.gpuUniverseSizeYEdit->setText(QString::number(value.y));
}

uint ComputationSettingsWidget::getNumBlocks() const
{
    return getUIntOrZero(ui.gpuNumBlocksEdit->text());
}

void ComputationSettingsWidget::setNumBlocks(uint value) const
{
    ui.gpuNumBlocksEdit->setText(QString::number(value));
}

uint ComputationSettingsWidget::getNumThreadsPerBlock() const
{
    return getUIntOrZero(ui.gpuNumThreadsPerBlockEdit->text());
}

void ComputationSettingsWidget::setNumThreadsPerBlock(uint value) const
{
    ui.gpuNumThreadsPerBlockEdit->setText(QString::number(value));
}

uint ComputationSettingsWidget::getMaxClusters() const
{
    return getUIntOrZero(ui.gpuMaxClustersEdit->text());
}

void ComputationSettingsWidget::setMaxClusters(uint value) const
{
    ui.gpuMaxClustersEdit->setText(QString::number(value));
}

uint ComputationSettingsWidget::getMaxCells() const
{
    return getUIntOrZero(ui.gpuMaxCellsEdit->text());
}

void ComputationSettingsWidget::setMaxCells(uint value) const
{
    ui.gpuMaxCellsEdit->setText(QString::number(value));
}

uint ComputationSettingsWidget::getMaxTokens() const
{
    return getUIntOrZero(ui.gpuMaxTokensEdit->text());
}

void ComputationSettingsWidget::setMaxTokens(uint value) const
{
    ui.gpuMaxTokensEdit->setText(QString::number(value));
}

uint ComputationSettingsWidget::getMaxParticles() const
{
    return getUIntOrZero(ui.gpuMaxParticlesEdit->text());
}

void ComputationSettingsWidget::setMaxParticles(uint value) const
{
    ui.gpuMaxParticlesEdit->setText(QString::number(value));
}

uint ComputationSettingsWidget::getDynamicMemorySize() const
{
    return getUIntOrZero(ui.gpuDynamicMemorySizeEdit->text());
}

void ComputationSettingsWidget::setDynamicMemorySize(uint value) const
{
    ui.gpuDynamicMemorySizeEdit->setText(QString::number(value));
}

void ComputationSettingsWidget::saveSettings()
{
    GuiSettings::setSettingsValue(Const::GpuNumBlocksKey, getNumBlocks());
    GuiSettings::setSettingsValue(Const::GpuNumThreadsPerBlockKey, getNumThreadsPerBlock());
    GuiSettings::setSettingsValue(Const::GpuMaxClustersKey, getMaxClusters());
    GuiSettings::setSettingsValue(Const::GpuMaxCellsKey, getMaxCells());
    GuiSettings::setSettingsValue(Const::GpuMaxTokensKey, getMaxTokens());
    GuiSettings::setSettingsValue(Const::GpuMaxParticlesKey, getMaxParticles());
    GuiSettings::setSettingsValue(Const::GpuDynamicMemorySizeKey, getDynamicMemorySize());
}
