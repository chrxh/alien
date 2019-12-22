#pragma once

#include <QWidget>

#include "Definitions.h"
#include "ui_ComputationSettingsWidget.h"


class ComputationSettingsWidget : public QWidget {
	Q_OBJECT

public:
	ComputationSettingsWidget(QWidget * parent = nullptr);
	~ComputationSettingsWidget() = default;

    IntVector2D getUniverseSize() const;
    void setUniverseSize(IntVector2D const& value) const;

    uint getNumBlocks() const;
    void setNumBlocks(uint value) const;

    uint getNumThreadsPerBlock() const;
    void setNumThreadsPerBlock(uint value) const;

    uint getMaxClusters() const;
    void setMaxClusters(uint value) const;

    uint getMaxCells() const;
    void setMaxCells(uint value) const;

    uint getMaxTokens() const;
    void setMaxTokens(uint value) const;

    uint getMaxParticles() const;
    void setMaxParticles(uint value) const;

    uint getDynamicMemorySize() const;
    void setDynamicMemorySize(uint value) const;

    void saveSettings();

private:
	Ui::ComputationSettingsWidget ui;
};
