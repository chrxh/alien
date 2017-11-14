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

	void init(IntVector2D const& upperLeftPosition, DataManipulator* manipulator, SimulationContext* context);

	DataEditContext* getContext() const;

	void notificationFromCellTab();
	void notificationFromClusterTab();
	void notificationFromParticleTab();
	void notificationFromMetadataTab();
	void notificationFromCellComputerTab();
	void notificationFromSymbolTab();

private:
	Q_SLOT void onShow(bool visible);
	Q_SLOT void notificationFromManipulator(set<DataManipulator::Receiver> const& targets);

	void switchToCellEditor(CellDescription const& cell);

	DataEditModel* _model = nullptr;
	DataEditorView* _view = nullptr;
	DataManipulator* _manipulator = nullptr;
	DataEditContext* _context = nullptr;
	SimulationParameters* _parameters = nullptr;
	SymbolTable* _symbolTable = nullptr;
};