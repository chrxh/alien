#pragma once
#include <QDialog>
#include "ui_ComputationSettingsDialog.h"

#include "Gui/Definitions.h"

class ComputationSettingsDialog
	: public QDialog
{
	Q_OBJECT
public:
	ComputationSettingsDialog(SimulationConfig const& config, QWidget * parent = nullptr);
	virtual ~ComputationSettingsDialog() = default;

    IntVector2D getUniverseSize() const;
    uint getNumBlocks() const;
    uint getNumThreadsPerBlock() const;
    uint getMaxClusters() const;
    uint getMaxCells() const;
    uint getMaxTokens() const;
    uint getMaxParticles() const;
    uint getDynamicMemorySize() const;
    bool isExtrapolateContent() const;

private:
	Q_SLOT void okClicked();

	Ui::ComputationSettingsDialog ui;
	SimulationConfig _config;
};
