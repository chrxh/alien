#pragma once

#include <QWidget>

#include "Gui/Definitions.h"
#include "Gui/DataController.h"

class DataEditController
	: public QObject
{
	Q_OBJECT
public:
	DataEditController(QWidget *parent = nullptr);
	virtual ~DataEditController() = default;

	void init(IntVector2D const& upperLeftPosition, Notifier* notifier, DataController* manipulator, SimulationContext* context);

	DataEditContext* getContext() const;

	void notificationFromCellTab();
	void notificationFromClusterTab();
	void notificationFromParticleTab();
	void notificationFromMetadataTab();
	void notificationFromCellComputerTab();
	void notificationFromSymbolTab();
	void notificationFromTokenTab();

private:
	Q_SLOT void onShow(bool visible);
	Q_SLOT void receivedExternalNotifications(set<Receiver> const& targets, UpdateDescription update);

	void switchToCellEditor(CellDescription const& cell, UpdateDescription update = UpdateDescription::All);

	list<QMetaObject::Connection> _connections;
	Notifier* _notifier = nullptr;
	DataEditModel* _model = nullptr;
	DataEditView* _view = nullptr;
	DataController* _manipulator = nullptr;
	DataEditContext* _context = nullptr;
	SimulationParameters const* _parameters = nullptr;
	SymbolTable const* _symbolTable = nullptr;
};