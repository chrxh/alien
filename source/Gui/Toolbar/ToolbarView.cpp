#include <QPushButton>
#include <QToolBar>

#include "Gui/Settings.h"

#include "Gui/Actions/ActionHolder.h"
#include "ToolbarView.h"
#include "ToolbarController.h"

ToolbarView::ToolbarView(QWidget * parent)
	: QWidget(parent)
{
	ui.setupUi(this);
	setStyleSheet(Const::ButtonStyleSheet);
}

void ToolbarView::init(IntVector2D const & upperLeftPosition, ActionHolder* actions, ToolbarController* controller)
{
	_controller = controller;
	_actions = actions;
	setGeometry(upperLeftPosition.x, upperLeftPosition.y, width(), height());

	for (auto const& connection : _connections) {
		disconnect(connection);
	}

	connectActionToButton(_actions->actionNewCell, ui.newCellButton);
	connectActionToButton(_actions->actionNewParticle, ui.newParticleButton);
	connectActionToButton(_actions->actionDeleteSel, ui.delSelectionButton);
	connectActionToButton(_actions->actionDeleteCol, ui.delCollectionButton);
	connectActionToButton(_actions->actionNewToken, ui.newTokenButton);
	connectActionToButton(_actions->actionDeleteToken, ui.delTokenButton);
	connectActionToButton(_actions->actionShowCellInfo, ui.showCellInfoButton);
	connectActionToButton(_actions->actionCenterSelection, ui.centerSelectionButton);
}

//note: workaround since QToolButton::setDefaultAction does not function as wished
//		(VisualEditor is not updated properly after pressing delete cell button)
void ToolbarView::connectActionToButton(QAction *& action, QToolButton *& button)
{
	button->setEnabled(action->isEnabled());
	button->setIcon(action->icon());
	button->setToolTip(action->toolTip());
	_connections.push_back(connect(action, &QAction::changed, [&]() {
		button->setEnabled(action->isEnabled());
	}));
	_connections.push_back(connect(action, &QAction::toggled, button, &QToolButton::setChecked));
	_connections.push_back(connect(button, &QToolButton::clicked, action, &QAction::trigger));
}
