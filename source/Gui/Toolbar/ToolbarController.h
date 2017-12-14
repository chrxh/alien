#pragma once
#include <QWidget>

#include "Gui/Definitions.h"
#include "Gui/DataController.h"

class ToolbarController
	: public QObject
{
	Q_OBJECT
public:
	ToolbarController(QWidget * parent = nullptr);
	virtual ~ToolbarController() = default;

	void init(IntVector2D const& upperLeftPosition, Notifier* notifier, DataController* manipulator, const SimulationContext* context);

	ToolbarContext* getContext() const;

	void onRequestCell();
	void onRequestParticle();
	void onDeleteSelection();
	void onDeleteExtendedSelection();
	void onRequestToken();
	void onDeleteToken();
	void onToggleCellInfo(bool showInfo);

private:
	Q_SLOT void onShow(bool visible);
	Q_SLOT void receivedNotifications(set<Receiver> const& targets);

	list<QMetaObject::Connection> _connections;
	Notifier* _notifier = nullptr;
	ToolbarContext* _context = nullptr;
	ToolbarView* _view = nullptr;
	ToolbarModel* _model = nullptr;
	DataController* _manipulator = nullptr;
	const SimulationParameters* _parameters = nullptr;
};
