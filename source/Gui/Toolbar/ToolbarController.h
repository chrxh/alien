#pragma once
#include <QWidget>

#include "Gui/Definitions.h"
#include "Gui/DataRepository.h"

class ToolbarController
	: public QObject
{
	Q_OBJECT
public:
	ToolbarController(QWidget * parent = nullptr);
	virtual ~ToolbarController() = default;

	void init(IntVector2D const& upperLeftPosition, Notifier* notifier, DataRepository* manipulator
		, SimulationContext const* context, ActionHolder* actions);

	ToolbarContext* getContext() const;

private:
	Q_SLOT void onShow(bool visible);

	Notifier* _notifier = nullptr;
	ToolbarContext* _context = nullptr;
	ToolbarView* _view = nullptr;
	DataRepository* _repository = nullptr;
	SimulationParameters const* _parameters = nullptr;
	ActionHolder* _actions = nullptr;
};
