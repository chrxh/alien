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
    uint getNumBlocks() const;
    uint getNumThreadsPerBlock() const;
    uint getMaxClusters() const;
    uint getMaxCells() const;
    uint getMaxTokens() const;
    uint getMaxParticles() const;
    uint getDynamicMemorySize() const;

    void saveSettings();

private:
	Ui::ComputationSettingsWidget ui;
};
