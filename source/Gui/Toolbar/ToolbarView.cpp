#include <QPushButton>

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

	ui.newCellButton->setDefaultAction(actions->actionNewCell);
	ui.newParticleButton->setDefaultAction(actions->actionNewParticle);
	ui.delSelectionButton->setDefaultAction(actions->actionDeleteSel);
	ui.delCollectionButton->setDefaultAction(actions->actionDeleteCol);
	ui.newTokenButton->setDefaultAction(actions->actionNewToken);
	ui.delTokenButton->setDefaultAction(actions->actionDeleteToken);
	ui.showCellInfoButton->setDefaultAction(actions->actionShowCellInfo);
	/*
	for (auto const& connection : _connections) {
		disconnect(connection);
	}
	_connections.push_back(connect(ui.requestCellButton, &QPushButton::clicked, [this]() { _controller->onRequestCell(); }));
	_connections.push_back(connect(ui.requestParticleButton, &QPushButton::clicked, [this]() { _controller->onRequestParticle(); }));
	_connections.push_back(connect(ui.delSelectionButton, &QPushButton::clicked, [this]() { _controller->onDeleteSelection(); }));
	_connections.push_back(connect(ui.delExtendedSelectionButton, &QPushButton::clicked, [this]() { _controller->onDeleteExtendedSelection(); }));
	_connections.push_back(connect(ui.requestTokenButton, &QPushButton::clicked, [this]() { _controller->onRequestToken(); }));
	_connections.push_back(connect(ui.delTokenButton, &QPushButton::clicked, [this]() { _controller->onDeleteToken(); }));
	_connections.push_back(connect(ui.showCellInfoButton, &QPushButton::toggled, [this](bool checked) {
		if (checked) {
			ui.showCellInfoButton->setIcon(QIcon("://Icons/info_on.png"));
		}
		else {
			ui.showCellInfoButton->setIcon(QIcon("://Icons/info_off.png"));
		}
		_controller->onToggleCellInfo(checked);
	}));

	_connections.push_back(connect(_actions->actionNewCell, &QAction::changed, [&]() {
		ui.requestCellButton->setEnabled(_actions->actionNewCell->isEnabled());
	}));
	_connections.push_back(connect(_actions->actionNewParticle, &QAction::changed, [&]() {
		ui.requestParticleButton->setEnabled(_actions->actionNewParticle->isEnabled());
	}));
	_connections.push_back(connect(_actions->actionDeleteSel, &QAction::changed, [&]() {
		ui.delSelectionButton->setEnabled(_actions->actionDeleteSel->isEnabled());
	}));
	_connections.push_back(connect(_actions->actionDeleteCol, &QAction::changed, [&]() {
		ui.delExtendedSelectionButton->setEnabled(_actions->actionDeleteCol->isEnabled());
	}));
	_connections.push_back(connect(_actions->actionNewToken, &QAction::changed, [&]() {
		ui.requestTokenButton->setEnabled(_actions->actionNewToken->isEnabled());
	}));
	_connections.push_back(connect(_actions->actionDeleteToken, &QAction::changed, [&]() {
		ui.delTokenButton->setEnabled(_actions->actionDeleteToken->isEnabled());
	}));
	_connections.push_back(connect(_actions->actionShowCellInfo, &QAction::changed, [&]() {
		ui.showCellInfoButton->setEnabled(_actions->actionShowCellInfo->isEnabled());
	}));

	ui.showCellInfoButton->setChecked(false);
*/
}
