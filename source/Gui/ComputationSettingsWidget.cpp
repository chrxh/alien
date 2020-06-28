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
    ui.gpuMetadataDynamicMemorySizeEdit->setText(StringHelper::toString(
        GuiSettings::getSettingsValue(Const::GpuMetadataDynamicMemorySizeKey, Const::GpuDynamicMemorySizeDefault)));
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

CudaConstants ComputationSettingsWidget::getCudaConstants() const
{
    CudaConstants result;
    result.NUM_BLOCKS = getUIntOrZero(ui.gpuNumBlocksEdit->text());
    result.NUM_THREADS_PER_BLOCK = getUIntOrZero(ui.gpuNumThreadsPerBlockEdit->text());
    result.MAX_CLUSTERS = getUIntOrZero(ui.gpuMaxClustersEdit->text());
    result.MAX_CLUSTERPOINTERS = result.MAX_CLUSTERS * 10;
    result.MAX_CELLS = getUIntOrZero(ui.gpuMaxCellsEdit->text());
    result.MAX_CELLPOINTERS = result.MAX_CELLS * 10;
    result.MAX_TOKENS = getUIntOrZero(ui.gpuMaxTokensEdit->text());
    result.MAX_TOKENPOINTERS = result.MAX_TOKENS * 10;
    result.MAX_PARTICLES = getUIntOrZero(ui.gpuMaxParticlesEdit->text());
    result.MAX_PARTICLEPOINTERS = result.MAX_PARTICLES * 10;
    result.DYNAMIC_MEMORY_SIZE = getUIntOrZero(ui.gpuDynamicMemorySizeEdit->text());
    result.METADATA_DYNAMIC_MEMORY_SIZE = getUIntOrZero(ui.gpuMetadataDynamicMemorySizeEdit->text());
    return result;
}

void ComputationSettingsWidget::setCudaConstants(CudaConstants const & value)
{
    ui.gpuNumBlocksEdit->setText(QString::number(value.NUM_BLOCKS));
    ui.gpuNumThreadsPerBlockEdit->setText(QString::number(value.NUM_THREADS_PER_BLOCK));
    ui.gpuMaxClustersEdit->setText(QString::number(value.MAX_CLUSTERS));
    ui.gpuMaxCellsEdit->setText(QString::number(value.MAX_CELLS));
    ui.gpuMaxTokensEdit->setText(QString::number(value.MAX_TOKENS));
    ui.gpuMaxParticlesEdit->setText(QString::number(value.MAX_PARTICLES));
    ui.gpuDynamicMemorySizeEdit->setText(QString::number(value.DYNAMIC_MEMORY_SIZE));
    ui.gpuMetadataDynamicMemorySizeEdit->setText(QString::number(value.METADATA_DYNAMIC_MEMORY_SIZE));
}

void ComputationSettingsWidget::saveSettings()
{
    auto cudaConstants = getCudaConstants();
    GuiSettings::setSettingsValue(Const::GpuUniverseSizeXKey, getUniverseSize().x);
    GuiSettings::setSettingsValue(Const::GpuUniverseSizeYKey, getUniverseSize().y);
    GuiSettings::setSettingsValue(Const::GpuNumBlocksKey, cudaConstants.NUM_BLOCKS);
    GuiSettings::setSettingsValue(Const::GpuNumThreadsPerBlockKey, cudaConstants.NUM_THREADS_PER_BLOCK);
    GuiSettings::setSettingsValue(Const::GpuMaxClustersKey, cudaConstants.MAX_CLUSTERS);
    GuiSettings::setSettingsValue(Const::GpuMaxCellsKey, cudaConstants.MAX_CELLS);
    GuiSettings::setSettingsValue(Const::GpuMaxTokensKey, cudaConstants.MAX_TOKENS);
    GuiSettings::setSettingsValue(Const::GpuMaxParticlesKey, cudaConstants.MAX_PARTICLES);
    GuiSettings::setSettingsValue(Const::GpuDynamicMemorySizeKey, cudaConstants.DYNAMIC_MEMORY_SIZE);
    GuiSettings::setSettingsValue(Const::GpuMetadataDynamicMemorySizeKey, cudaConstants.METADATA_DYNAMIC_MEMORY_SIZE);
}
