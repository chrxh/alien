#pragma once

#include <QWidget>

#include "Gui/Definitions.h"
#include "Gui/DataManipulator.h"

class DataEditController
	: public QObject
{
	Q_OBJECT
public:
	DataEditController(QWidget *parent = nullptr);
	virtual ~DataEditController() = default;

	void init(IntVector2D const& upperLeftPosition, Notifier* notifier, DataManipulator* manipulator, SimulationContext* context);

	DataEditContext* getContext() const;

	void notificationFromCellTab();
	void notificationFromClusterTab();
	void notificationFromParticleTab();
	void notificationFromMetadataTab();
	void notificationFromCellComputerTab();
	void notificationFromSymbolTab();

private:
	Q_SLOT void onShow(bool visible);
	Q_SLOT void receivedExternalNotifications(set<Receiver> const& targets, UpdateDescription update);

	void switchToCellEditor(CellDescription const& cell, UpdateDescription update = UpdateDescription::All);

	Notifier* _notifier = nullptr;
	DataEditModel* _model = nullptr;
	DataEditorView* _view = nullptr;
	DataManipulator* _manipulator = nullptr;
	DataEditContext* _context = nullptr;
	SimulationParameters* _parameters = nullptr;
	SymbolTable* _symbolTable = nullptr;
};