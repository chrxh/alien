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
	setStyleSheet(GuiSettings::ButtonStyleSheet);

}

void ToolbarView::init(IntVector2D const & upperLeftPosition, ActionHolder* actions, ToolbarController* controller)
{
	_controller = controller;
	_actions = actions;
	setGeometry(upperLeftPosition.x, upperLeftPosition.y, width(), height());

	connectActionToButton(_actions->actionNewCell, ui.newCellButton);
	connectActionToButton(_actions->actionNewParticle, ui.newParticleButton);
	connectActionToButton(_actions->actionDeleteSel, ui.delSelectionButton);
	connectActionToButton(_actions->actionDeleteCol, ui.delCollectionButton);
	connectActionToButton(_actions->actionNewToken, ui.newTokenButton);
	connectActionToButton(_actions->actionDeleteToken, ui.delTokenButton);
	connectActionToButton(_actions->actionShowCellInfo, ui.showCellInfoButton);
}

//note: workaround since QToolButton::setDefaultAction does not work as wished
//		(VisualEditor is not updated properly after pressing delete cell button)
void ToolbarView::connectActionToButton(QAction *& action, QToolButton *& button)
{
	button->setEnabled(action->isEnabled());
	button->setIcon(action->icon());
	connect(action, &QAction::changed, [&]() {
		button->setEnabled(action->isEnabled());
	});
	connect(action, &QAction::toggled, button, &QToolButton::setChecked);
	connect(button, &QToolButton::clicked, action, &QAction::trigger);
}
