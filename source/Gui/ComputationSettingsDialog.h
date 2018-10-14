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

	optional<uint> getMaxThreads() const;
	optional<IntVector2D> getGridSize() const;
	optional<IntVector2D> getUniverseSize() const;

private:
	void updateLabels();

	Q_SLOT void okClicked();

	Ui::ComputationSettingsDialog ui;
	SimulationConfig _config;
};
