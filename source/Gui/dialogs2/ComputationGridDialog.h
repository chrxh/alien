#pragma once
#include <QDialog>
#include "ui_ComputationGridDialog.h"

#include "Gui/Definitions.h"

class ComputationGridDialog
	: public QDialog
{
	Q_OBJECT
public:
	ComputationGridDialog(SimulationConfig const& config, QWidget * parent = nullptr);
	virtual ~ComputationGridDialog() = default;

	optional<uint> getMaxThreads() const;
	optional<IntVector2D> getGridSize() const;
	optional<IntVector2D> getUniverseSize() const;

private:
	void updateUniverseSize();

	Ui::ComputationGridDialog ui;
};
