#pragma once
#include <QWidget>

#include "Gui/Definitions.h"
#include "Gui/DataManipulator.h"

class ToolbarController
	: public QObject
{
	Q_OBJECT
public:
	ToolbarController(QWidget * parent = nullptr);
	virtual ~ToolbarController() = default;

	void init(IntVector2D const& upperLeftPosition, DataManipulator* manipulator, const SimulationContext* context);

	ToolbarContext* getContext() const;

	void onRequestCell();
	void onRequestParticle();
	void onDeleteSelection();
	void onDeleteExtendedSelection();
	void onRequestToken();

private:
	Q_SLOT void onShow(bool visible);
	Q_SLOT void notificationFromManipulator(set<DataManipulator::Receiver> const& targets);

	ToolbarContext* _context = nullptr;
	ToolbarView* _view = nullptr;
	ToolbarModel* _model = nullptr;
	DataManipulator* _manipulator = nullptr;
	const SimulationParameters* _parameters = nullptr;
};
